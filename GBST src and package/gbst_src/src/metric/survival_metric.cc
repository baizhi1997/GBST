/*!
 * Copyright 2015 by Contributors
 * \file rank_metric.cc
 * \brief prediction rank based metrics.
 * \author Kailong Chen, Tianqi Chen
 */
#include <rabit/rabit.h>
#include <xgboost/metric.h>
#include <dmlc/registry.h>
#include <cmath>

#include <vector>

#include "xgboost/host_device_vector.h"
#include "../common/math.h"

namespace {

/*
 * Adapter to access instance weights.
 *
 *  - For ranking task, weights are per-group
 *  - For binary classification task, weights are per-instance
 *
 * WeightPolicy::GetWeightOfInstance() :
 *   get weight associated with an individual instance, using index into
 *   `info.weights`
 * WeightPolicy::GetWeightOfSortedRecord() :
 *   get weight associated with an individual instance, using index into
 *   sorted records `rec` (in ascending order of predicted labels). `rec` is
 *   of type PredIndPairContainer
 */

using PredIndPairContainer
  = std::vector<std::pair<xgboost::bst_float, unsigned>>;

class PerInstanceWeightPolicy {
 public:
  inline static xgboost::bst_float
  GetWeightOfInstance(const xgboost::MetaInfo& info,
                      unsigned instance_id, unsigned group_id) {
    return info.GetWeight(instance_id);
  }
  inline static xgboost::bst_float
  GetWeightOfSortedRecord(const xgboost::MetaInfo& info,
                          const PredIndPairContainer& rec,
                          unsigned record_id, unsigned group_id) {
    return info.GetWeight(rec[record_id].second);
  }
};

}  // anonymous namespace

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(survival_metric);


/*! \brief Area Under Curve, for both classification and rank */
struct EvalAuc : public Metric {
 private:
  template <typename WeightPolicy>
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    const size_t nclass = preds.Size() / info.labels_.Size();
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.empty() ? tgptr : info.group_ptr_;
    CHECK_EQ(gptr.back(), info.labels_.Size())
        << "EvalAuc: group structure must match number of prediction";
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum of all AUC's across all query groups
    double sum_auc = 0.0;
    int auc_error = 0;
    // each thread takes a local rec
    std::vector<std::pair<bst_float, unsigned>> rec;
    const auto& labels = info.labels_.HostVector();
    const std::vector<bst_float>& h_preds = preds.HostVector();
    for (bst_omp_uint group_id = 0; group_id < ngroup; ++group_id) {
      // form a rec for EACH timestep.
      double group_auc = 0.0;
      int group_auc_error = 0;
      rec.clear();
      for (unsigned j = gptr[group_id]; j < gptr[group_id + 1]; ++j) {
          rec.emplace_back(1, j);
        };
      for (bst_omp_uint timestep=0; timestep < nclass; ++timestep){
        for (int instance=0; instance<info.labels_.Size(); ++instance){
          bst_float hazard = 1.0f / (1.0f + expf(-h_preds[instance*nclass + timestep]));
          rec[instance].first *= hazard;
        }
        std::vector<std::pair<bst_float, unsigned>> rec1(rec);
        XGBOOST_PARALLEL_STABLE_SORT(rec1.begin(), rec1.end(), common::CmpFirst);
        // calculate AUC
        double sum_pospair = 0.0;
        double sum_npos = 0.0, sum_nneg = 0.0, buf_pos = 0.0, buf_neg = 0.0;
        for (size_t j = 0; j < rec1.size(); ++j) {
          const bst_float wt
            = WeightPolicy::GetWeightOfSortedRecord(info, rec, j, group_id);
          bst_float ctr = 1.0;
          if (labels[rec1[j].second]>=timestep) ctr = 0.0;
          // keep bucketing predictions in same bucket
          if (j != 0 && rec1[j].first != rec1[j - 1].first) {
            sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
            sum_npos += buf_pos;
            sum_nneg += buf_neg;
            buf_neg = buf_pos = 0.0f;
          }
          buf_pos += ctr * wt;
          buf_neg += (1.0f - ctr) * wt;
        }
        sum_pospair += buf_neg * (sum_npos + buf_pos * 0.5);
        sum_npos += buf_pos;
        sum_nneg += buf_neg;
        // check weird conditions
        if (sum_npos <= 0.0 || sum_nneg <= 0.0) {
          group_auc_error += 1;
          continue;
        }
        // this is the AUC
        sum_auc += sum_pospair / (sum_npos * sum_nneg);
      }
      if (group_auc_error==static_cast<int>(nclass)){
        auc_error+=1;
        continue;
      }
      else sum_auc /= (static_cast<int>(nclass) - group_auc_error);
    }

    // Report average AUC across all groups
    // In distributed mode, workers which only contains pos or neg samples
    // will be ignored when aggregate AUC.
    bst_float dat[2] = {0.0f, 0.0f};
    if (auc_error < static_cast<int>(ngroup)) {
      dat[0] = static_cast<bst_float>(sum_auc);
      dat[1] = static_cast<bst_float>(static_cast<int>(ngroup) - auc_error);
    }
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    CHECK_GT(dat[1], 0.0f)
      << "AUC: the dataset only contains pos or neg samples";
    return dat[0] / dat[1];
  }

 public:
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    return Eval<PerInstanceWeightPolicy>(preds, info, distributed);
    }
  const char* Name() const override {
    return "auc";
  }
};

XGBOOST_REGISTER_METRIC(Auc, "survivalauc")
.describe("Area under curve for survival tasks.")
.set_body([](const char* param) { return new EvalAuc(); });

}  // namespace metric
}  // namespace xgboost