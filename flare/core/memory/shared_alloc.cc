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

#include <flare/core.h>
#include <iomanip>

namespace flare {
namespace detail {

#ifdef FLARE_ENABLE_DEBUG
bool SharedAllocationRecord<void, void>::is_sane(
    SharedAllocationRecord<void, void>* arg_record) {
  SharedAllocationRecord* const root =
      arg_record ? arg_record->m_root : nullptr;

  bool ok = root != nullptr && root->use_count() == 0;

  if (ok) {
    SharedAllocationRecord* root_next             = nullptr;
    static constexpr SharedAllocationRecord* zero = nullptr;
    // Lock the list:
    while ((root_next = flare::atomic_exchange(&root->m_next, zero)) ==
           nullptr)
      ;

    for (SharedAllocationRecord* rec = root_next; ok && rec != root;
         rec                         = rec->m_next) {
      const bool ok_non_null =
          rec && rec->m_prev && (rec == root || rec->m_next);
      const bool ok_root = ok_non_null && rec->m_root == root;
      const bool ok_prev_next =
          ok_non_null &&
          (rec->m_prev != root ? rec->m_prev->m_next == rec : root_next == rec);
      const bool ok_next_prev = ok_non_null && rec->m_next->m_prev == rec;
      const bool ok_count     = ok_non_null && 0 <= rec->use_count();

      ok = ok_root && ok_prev_next && ok_next_prev && ok_count;

      if (!ok) {
        // Formatting dependent on sizeof(uintptr_t)
        const char* format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string =
              "flare::detail::SharedAllocationRecord failed is_sane: "
              "rec(0x%.12lx){ m_count(%d) m_root(0x%.12lx) m_next(0x%.12lx) "
              "m_prev(0x%.12lx) m_next->m_prev(0x%.12lx) "
              "m_prev->m_next(0x%.12lx) }\n";
        } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string =
              "flare::detail::SharedAllocationRecord failed is_sane: "
              "rec(0x%.12llx){ m_count(%d) m_root(0x%.12llx) m_next(0x%.12llx) "
              "m_prev(0x%.12llx) m_next->m_prev(0x%.12llx) "
              "m_prev->m_next(0x%.12llx) }\n";
        }

        fprintf(stderr, format_string, reinterpret_cast<uintptr_t>(rec),
                rec->use_count(), reinterpret_cast<uintptr_t>(rec->m_root),
                reinterpret_cast<uintptr_t>(rec->m_next),
                reinterpret_cast<uintptr_t>(rec->m_prev),
                reinterpret_cast<uintptr_t>(
                    rec->m_next != nullptr ? rec->m_next->m_prev : nullptr),
                reinterpret_cast<uintptr_t>(rec->m_prev != rec->m_root
                                                ? rec->m_prev->m_next
                                                : root_next));
      }
    }

    if (nullptr != flare::atomic_exchange(&root->m_next, root_next)) {
      flare::detail::throw_runtime_exception(
          "flare::detail::SharedAllocationRecord failed is_sane unlocking");
    }
  }
  return ok;
}

#else

bool SharedAllocationRecord<void, void>::is_sane(
    SharedAllocationRecord<void, void>*) {
  flare::detail::throw_runtime_exception(
      "flare::detail::SharedAllocationRecord::is_sane only works with "
      "FLARE_ENABLE_DEBUG enabled");
  return false;
}
#endif  //#ifdef FLARE_ENABLE_DEBUG

#ifdef FLARE_ENABLE_DEBUG
SharedAllocationRecord<void, void>* SharedAllocationRecord<void, void>::find(
    SharedAllocationRecord<void, void>* const arg_root,
    void* const arg_data_ptr) {
  SharedAllocationRecord* root_next             = nullptr;
  static constexpr SharedAllocationRecord* zero = nullptr;

  // Lock the list:
  while ((root_next = flare::atomic_exchange(&arg_root->m_next, zero)) ==
         nullptr)
    ;

  // Iterate searching for the record with this data pointer

  SharedAllocationRecord* r = root_next;

  while ((r != arg_root) && (r->data() != arg_data_ptr)) {
    r = r->m_next;
  }

  if (r == arg_root) {
    r = nullptr;
  }

  if (nullptr != flare::atomic_exchange(&arg_root->m_next, root_next)) {
    flare::detail::throw_runtime_exception(
        "flare::detail::SharedAllocationRecord failed locking/unlocking");
  }
  return r;
}
#else
SharedAllocationRecord<void, void>* SharedAllocationRecord<void, void>::find(
    SharedAllocationRecord<void, void>* const, void* const) {
  flare::detail::throw_runtime_exception(
      "flare::detail::SharedAllocationRecord::find only works with "
      "FLARE_ENABLE_DEBUG "
      "enabled");
  return nullptr;
}
#endif

/**\brief  Construct and insert into 'arg_root' tracking set.
 *         use_count is zero.
 */
SharedAllocationRecord<void, void>::SharedAllocationRecord(
#ifdef FLARE_ENABLE_DEBUG
    SharedAllocationRecord<void, void>* arg_root,
#endif
    SharedAllocationHeader* arg_alloc_ptr, size_t arg_alloc_size,
    SharedAllocationRecord<void, void>::function_type arg_dealloc,
    const std::string& label)
    : m_alloc_ptr(arg_alloc_ptr),
      m_alloc_size(arg_alloc_size),
      m_dealloc(arg_dealloc)
#ifdef FLARE_ENABLE_DEBUG
      ,
      m_root(arg_root),
      m_prev(nullptr),
      m_next(nullptr)
#endif
      ,
      m_count(0),
      m_label(label) {
  if (nullptr != arg_alloc_ptr) {
#ifdef FLARE_ENABLE_DEBUG
    // Insert into the root double-linked list for tracking
    //
    // before:  arg_root->m_next == next ; next->m_prev == arg_root
    // after:   arg_root->m_next == this ; this->m_prev == arg_root ;
    //              this->m_next == next ; next->m_prev == this

    m_prev                                        = m_root;
    static constexpr SharedAllocationRecord* zero = nullptr;

    // Read root->m_next and lock by setting to nullptr
    while ((m_next = flare::atomic_exchange(&m_root->m_next, zero)) == nullptr)
      ;

    m_next->m_prev = this;

    // memory fence before completing insertion into linked list
    flare::memory_fence();

    if (nullptr != flare::atomic_exchange(&m_root->m_next, this)) {
      flare::detail::throw_runtime_exception(
          "flare::detail::SharedAllocationRecord failed locking/unlocking");
    }
#endif

  } else {
    flare::detail::throw_runtime_exception(
        "flare::detail::SharedAllocationRecord given nullptr allocation");
  }
}

void SharedAllocationRecord<void, void>::increment(
    SharedAllocationRecord<void, void>* arg_record) {
  const int old_count = flare::atomic_fetch_add(&arg_record->m_count, 1);

  if (old_count < 0) {  // Error
    flare::detail::throw_runtime_exception(
        "flare::detail::SharedAllocationRecord failed increment");
  }
}

SharedAllocationRecord<void, void>* SharedAllocationRecord<
    void, void>::decrement(SharedAllocationRecord<void, void>* arg_record) {
  const int old_count = flare::atomic_fetch_sub(&arg_record->m_count, 1);

  if (old_count == 1) {
    if (is_finalized()) {
      std::stringstream ss;
      ss << "flare allocation \"";
      ss << arg_record->get_label();
      ss << "\" is being deallocated after flare::finalize was called\n";
      auto s = ss.str();
      flare::detail::throw_runtime_exception(s);
    }

#ifdef FLARE_ENABLE_DEBUG
    // before:  arg_record->m_prev->m_next == arg_record  &&
    //          arg_record->m_next->m_prev == arg_record
    //
    // after:   arg_record->m_prev->m_next == arg_record->m_next  &&
    //          arg_record->m_next->m_prev == arg_record->m_prev

    SharedAllocationRecord* root_next             = nullptr;
    static constexpr SharedAllocationRecord* zero = nullptr;

    // Lock the list:
    while ((root_next = flare::atomic_exchange(&arg_record->m_root->m_next,
                                                zero)) == nullptr)
      ;
    // We need a memory_fence() here so that the following update
    // is properly sequenced
    flare::memory_fence();

    arg_record->m_next->m_prev = arg_record->m_prev;

    if (root_next != arg_record) {
      arg_record->m_prev->m_next = arg_record->m_next;
    } else {
      // before:  arg_record->m_root == arg_record->m_prev
      // after:   arg_record->m_root == arg_record->m_next
      root_next = arg_record->m_next;
    }

    flare::memory_fence();

    // Unlock the list:
    if (nullptr !=
        flare::atomic_exchange(&arg_record->m_root->m_next, root_next)) {
      flare::detail::throw_runtime_exception(
          "flare::detail::SharedAllocationRecord failed decrement unlocking");
    }

    arg_record->m_next = nullptr;
    arg_record->m_prev = nullptr;
#endif

    function_type d = arg_record->m_dealloc;
    (*d)(arg_record);
    arg_record = nullptr;
  } else if (old_count < 1) {  // Error
    fprintf(stderr,
            "flare::detail::SharedAllocationRecord '%s' failed decrement count "
            "= %d\n",
            arg_record->m_alloc_ptr->m_label, old_count);
    fflush(stderr);
    flare::detail::throw_runtime_exception(
        "flare::detail::SharedAllocationRecord failed decrement count");
  }

  return arg_record;
}

#ifdef FLARE_ENABLE_DEBUG
void SharedAllocationRecord<void, void>::print_host_accessible_records(
    std::ostream& s, const char* const space_name,
    const SharedAllocationRecord* const root, const bool detail) {
  // Print every node except the root, which does not represent an actual
  // allocation.
  const SharedAllocationRecord<void, void>* r = root->m_next;

  std::ios_base::fmtflags saved_flags = s.flags();
#define FLARE_IMPL_PAD_HEX(ptr)                              \
  "0x" << std::hex << std::setw(12) << std::setfill('0') \
       << reinterpret_cast<uintptr_t>(ptr)
  if (detail) {
    while (r != root) {
      s << space_name << " addr( " << FLARE_IMPL_PAD_HEX(r) << " ) list ( "
        << FLARE_IMPL_PAD_HEX(r->m_prev) << ' ' << FLARE_IMPL_PAD_HEX(r->m_next)
        << " ) extent[ " << FLARE_IMPL_PAD_HEX(r->m_alloc_ptr) << " + " << std::dec
        << std::setw(8) << r->m_alloc_size << " ] count(" << r->use_count()
        << ") dealloc(" << FLARE_IMPL_PAD_HEX(r->m_dealloc) << ") "
        << r->m_alloc_ptr->m_label << '\n';

      r = r->m_next;
    }
  } else {
    while (r != root) {
      s << space_name << " [ " << FLARE_IMPL_PAD_HEX(r->data()) << " + " << std::dec
        << r->size() << " ] " << r->m_alloc_ptr->m_label << '\n';
      r = r->m_next;
    }
  }
#undef FLARE_IMPL_PAD_HEX
  s.flags(saved_flags);
}
#else
void SharedAllocationRecord<void, void>::print_host_accessible_records(
    std::ostream&, const char* const, const SharedAllocationRecord* const,
    const bool) {
  flare::detail::throw_runtime_exception(
      "flare::detail::SharedAllocationRecord::print_host_accessible_records"
      " only works with FLARE_ENABLE_DEBUG enabled");
}
#endif

} /* namespace detail */
} /* namespace flare */
