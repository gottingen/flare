// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef FLARE_ERROR_REPORTER_H_
#define FLARE_ERROR_REPORTER_H_

#include <vector>
#include <flare/core.h>
#include <flare/core/tensor/view.h>
#include <flare/dual_view.h>

namespace flare {
namespace experimental {

template <typename ReportType, typename DeviceType>
class ErrorReporter {
 public:
  using report_type     = ReportType;
  using device_type     = DeviceType;
  using execution_space = typename device_type::execution_space;

  ErrorReporter(int max_results)
      : m_numReportsAttempted(""),
        m_reports("", max_results),
        m_reporters("", max_results) {
    clear();
  }

  int getCapacity() const { return m_reports.h_view.extent(0); }

  int getNumReports();

  int getNumReportAttempts();

  void getReports(std::vector<int> &reporters_out,
                  std::vector<report_type> &reports_out);
  void getReports(
      typename flare::View<int *,
                            typename DeviceType::execution_space>::HostMirror
          &reporters_out,
      typename flare::View<report_type *,
                            typename DeviceType::execution_space>::HostMirror
          &reports_out);

  void clear();

  void resize(const size_t new_size);

  bool full() { return (getNumReportAttempts() >= getCapacity()); }

  FLARE_INLINE_FUNCTION
  bool add_report(int reporter_id, report_type report) const {
    int idx = flare::atomic_fetch_add(&m_numReportsAttempted(), 1);

    if (idx >= 0 && (idx < static_cast<int>(m_reports.d_view.extent(0)))) {
      m_reporters.d_view(idx) = reporter_id;
      m_reports.d_view(idx)   = report;
      return true;
    } else {
      return false;
    }
  }

 private:
  using reports_view_t     = flare::View<report_type *, device_type>;
  using reports_dualview_t = flare::DualView<report_type *, device_type>;

  using host_mirror_space = typename reports_dualview_t::host_mirror_space;
  flare::View<int, device_type> m_numReportsAttempted;
  reports_dualview_t m_reports;
  flare::DualView<int *, device_type> m_reporters;
};

template <typename ReportType, typename DeviceType>
inline int ErrorReporter<ReportType, DeviceType>::getNumReports() {
  int num_reports = 0;
  flare::deep_copy(num_reports, m_numReportsAttempted);
  if (num_reports > static_cast<int>(m_reports.h_view.extent(0))) {
    num_reports = m_reports.h_view.extent(0);
  }
  return num_reports;
}

template <typename ReportType, typename DeviceType>
inline int ErrorReporter<ReportType, DeviceType>::getNumReportAttempts() {
  int num_reports = 0;
  flare::deep_copy(num_reports, m_numReportsAttempted);
  return num_reports;
}

template <typename ReportType, typename DeviceType>
void ErrorReporter<ReportType, DeviceType>::getReports(
    std::vector<int> &reporters_out, std::vector<report_type> &reports_out) {
  int num_reports = getNumReports();
  reporters_out.clear();
  reporters_out.reserve(num_reports);
  reports_out.clear();
  reports_out.reserve(num_reports);

  if (num_reports > 0) {
    m_reports.template sync<host_mirror_space>();
    m_reporters.template sync<host_mirror_space>();

    for (int i = 0; i < num_reports; ++i) {
      reporters_out.push_back(m_reporters.h_view(i));
      reports_out.push_back(m_reports.h_view(i));
    }
  }
}

template <typename ReportType, typename DeviceType>
void ErrorReporter<ReportType, DeviceType>::getReports(
    typename flare::View<
        int *, typename DeviceType::execution_space>::HostMirror &reporters_out,
    typename flare::View<report_type *,
                          typename DeviceType::execution_space>::HostMirror
        &reports_out) {
  int num_reports = getNumReports();
  reporters_out   = typename flare::View<int *, DeviceType>::HostMirror(
      "ErrorReport::reporters_out", num_reports);
  reports_out = typename flare::View<report_type *, DeviceType>::HostMirror(
      "ErrorReport::reports_out", num_reports);

  if (num_reports > 0) {
    m_reports.template sync<host_mirror_space>();
    m_reporters.template sync<host_mirror_space>();

    for (int i = 0; i < num_reports; ++i) {
      reporters_out(i) = m_reporters.h_view(i);
      reports_out(i)   = m_reports.h_view(i);
    }
  }
}

template <typename ReportType, typename DeviceType>
void ErrorReporter<ReportType, DeviceType>::clear() {
  int num_reports = 0;
  flare::deep_copy(m_numReportsAttempted, num_reports);
  m_reports.template modify<execution_space>();
  m_reporters.template modify<execution_space>();
}

template <typename ReportType, typename DeviceType>
void ErrorReporter<ReportType, DeviceType>::resize(const size_t new_size) {
  m_reports.resize(new_size);
  m_reporters.resize(new_size);
  typename DeviceType::execution_space().fence(
      "flare::experimental::ErrorReporter::resize: fence after resizing");
}

}  // namespace experimental
}  // namespace flare

#endif  // FLARE_ERROR_REPORTER_H_
