/*******************************************************
 * Copyright (c) 2014, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef __cplusplus

#include <ostream>
#include <fly/defines.h>

namespace fly {

/// An Flare exception class
/// \ingroup arrayfire_class
class FLY_API exception : public std::exception
{
private:
    char m_msg[1024];
    fly_err m_err;
public:
    fly_err err() { return m_err; }
    exception();
    /// Creates a new fly::exception given a message. The error code is FLY_ERR_UNKNOWN
    exception(const char *msg);

    /// Creates a new exception with a formatted error message for a given file
    /// and line number in the source code.
    exception(const char *file, unsigned line, fly_err err);

    /// Creates a new fly::exception with a formatted error message for a given
    /// an error code, file and line number in the source code.
    exception(const char *msg, const char *file, unsigned line, fly_err err);
#if FLY_API_VERSION >= 33
    /// Creates a new exception given a message, function name, file name, line number and
    /// error code.
    exception(const char *msg, const char *func, const char *file, unsigned line, fly_err err);
#endif
    virtual ~exception() throw() {}
    /// Returns an error message for the exception in a string format
    virtual const char *what() const throw() { return m_msg; }

    /// Writes the exception to a stream
    friend inline std::ostream& operator<<(std::ostream &s, const exception &e)
    { return s << e.what(); }
};

}  // namespace fly

#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the last error message that occurred and its error message
///
/// \param[out] msg The message of the previous error
/// \param[out] len The number of characters in the msg object
FLY_API void fly_get_last_error(char **msg, dim_t *len);

/// Converts the fly_err error code to its string representation
///
/// \param[in] err The Flare error code
FLY_API const char *fly_err_to_string(const fly_err err);

#ifdef __cplusplus
}
#endif
