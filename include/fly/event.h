/*******************************************************
 * Copyright (c) 2019, Flare
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <fly/defines.h>

#if FLY_API_VERSION >= 37

/**
    Handle to an event object

    \ingroup event_api
*/
typedef void* fly_event;

#ifdef __cplusplus
namespace fly {

/**
    C++ RAII interface for manipulating events
    \ingroup arrayfire_class
    \ingroup event_api
*/
class FLY_API event {
    fly_event e_;

   public:
    /// Create a new event using the C fly_event handle
    event(fly_event e);
#if FLY_COMPILER_CXX_RVALUE_REFERENCES
    /// Move constructor
    event(event&& other);

    /// Move assignment operator
    event& operator=(event&& other);
#endif
    /// Create a new event object
    event();

    /// event Destructor
    ~event();

    /// Return the underlying C fly_event handle
    fly_event get() const;

    /// \brief Adds the event on the default Flare queue. Once this point
    ///        on the program is executed, the event is considered complete.
    void mark();

    /// \brief Block the Flare queue until this even has occurred
    void enqueue();

    /// \brief block the calling thread until this event has occurred
    void block() const;

   private:
    event& operator=(const event& other);
    event(const event& other);
};

}  // namespace fly
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   \brief Create a new \ref fly_event handle

   \param[in] eventHandle the input event handle

   \ingroup event_api
*/
FLY_API fly_err fly_create_event(fly_event* eventHandle);

/**
   \brief Release the \ref fly_event handle

   \param[in] eventHandle the input event handle

   \ingroup event_api
*/
FLY_API fly_err fly_delete_event(fly_event eventHandle);

/**
   marks the \ref fly_event on the active computation stream. If the \ref
   fly_event is enqueued/waited on later, any operations that are currently
   enqueued on the event stream will be completed before any events that are
   enqueued after the call to enqueue

   \param[in] eventHandle the input event handle

   \ingroup event_api
*/
FLY_API fly_err fly_mark_event(const fly_event eventHandle);

/**
   enqueues the \ref fly_event and all enqueued events on the active stream.
   All operations enqueued after a call to enqueue will not be executed
   until operations on the stream when mark was called are complete

   \param[in] eventHandle the input event handle

   \ingroup event_api
*/
FLY_API fly_err fly_enqueue_wait_event(const fly_event eventHandle);

/**
   blocks the calling thread on events until all events on the computation
   stream before mark was called are complete

   \param[in] eventHandle the input event handle

   \ingroup event_api
*/
FLY_API fly_err fly_block_event(const fly_event eventHandle);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // FLY_API_VERSION >= 37
