// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "common/unique_function.h"
#include "video_core/renderer_vulkan/pipeline_build.h"

namespace Vulkan {

template <typename Key, typename Pipeline>
class PipelinePrebuild {
    struct Entry {
        std::unique_ptr<EmissionBudget::Reservation> reservation;
        std::unique_ptr<Pipeline> pipeline;
        bool claimed{};
        // Written under PipelinePrebuild::mutex. `running` is set when the prebuild worker
        // has started the build, and `done` when it has finished (successfully or not).
        // A game thread that needs the pipeline waits only when the build is already
        // running: that is faster than rebuilding it from scratch, and it must not wait
        // for the whole queue ahead of it.
        bool running{};
        bool done{};
        bool cancelled{};
        std::condition_variable complete;
    };

public:
    using Build = Common::UniqueFunction<std::unique_ptr<Pipeline>>;

    PipelinePrebuild(size_t capacity, size_t capture_limit_, bool broken_parallel_compilation)
        : budget{broken_parallel_compilation ? 0 : capacity}, capture_limit{capture_limit_} {}

    ~PipelinePrebuild() {
        Stop();
    }

    bool Add(const Key& key, size_t capture_bytes, Build build) {
        if (started || stopped || entries.contains(key) ||
            capture_bytes > capture_limit - captured_bytes) {
            return false;
        }
        auto reservation = budget.TryAcquire(true);
        if (!reservation) {
            return false;
        }
        auto entry = std::make_shared<Entry>();
        entry->reservation = std::move(reservation);
        entries.emplace(key, entry);
        captured_bytes += capture_bytes;
        tasks.emplace_back([this, key, entry, build = std::move(build)]() mutable {
            bool run;
            {
                std::scoped_lock lock{mutex};
                run = !stopped && !entry->claimed;
            }
            std::unique_ptr<Pipeline> pipeline;
            if (run) {
                {
                    std::scoped_lock lock{mutex};
                    entry->running = true;
                    entry->complete.notify_all();
                }
                try {
                    pipeline = build();
                } catch (const std::exception& exception) {
                    LOG_ERROR(Render_Vulkan, "Cached pipeline prebuild failed: {}",
                              exception.what());
                } catch (...) {
                    LOG_ERROR(Render_Vulkan, "Cached pipeline prebuild failed");
                }
            }
            build = {};
            {
                std::scoped_lock lock{mutex};
                entry->done = true;
                if (!stopped && !entry->claimed && pipeline) {
                    entry->pipeline = std::move(pipeline);
                } else if (!entry->claimed) {
                    entries.erase(key);
                }
                entry->complete.notify_all();
            }
        });
        return true;
    }

    void Start(size_t threads) {
        if (started || stopped || tasks.empty() || threads == 0) {
            return;
        }
        started = true;
        workers.reset(new Common::ThreadWorker(threads, "VkPipelinePrebuild", {},
                                               Common::ThreadPlacement::Background));
        for (auto& task : tasks) {
            workers->QueueWork(std::move(task));
        }
        tasks.clear();
    }

    /// Removes the entry for `key` and returns its pipeline, building it if available.
    /// When `wait_in_flight` is set and the prebuild worker is already executing the build,
    /// wait for it and return its result instead of claiming the entry and discarding a
    /// build that may be nearly finished (the caller would otherwise rebuild the pipeline
    /// synchronously, duplicating the whole build). When the build has not started yet it is
    /// still claimed: waiting for it would mean waiting for every queued build before it.
    std::unique_ptr<Pipeline> Take(const Key& key, bool wait_in_flight = false) {
        std::shared_ptr<Entry> entry;
        {
            std::scoped_lock lock{mutex};
            const auto it = entries.find(key);
            if (it == entries.end()) {
                return nullptr;
            }
            entry = std::move(it->second);
            entries.erase(it);
            if (!wait_in_flight || entry->done || !entry->running) {
                entry->claimed = true;
                return std::move(entry->pipeline);
            }
        }
        // The build is running on a prebuild worker; wait for the result. The entry is not
        // in `entries` anymore, so Stop() cannot reach it from there: track it in `waiting`
        // too, so shutdown still wakes this waiter.
        {
            std::scoped_lock lock{mutex};
            waiting.push_back(entry);
        }
        std::unique_lock lock{mutex};
        entry->complete.wait(lock,
                             [&] { return entry->done || entry->claimed || entry->cancelled; });
        waiting.erase(std::remove(waiting.begin(), waiting.end(), entry), waiting.end());
        entry->claimed = true;
        return std::move(entry->pipeline);
    }

    void Stop() {
        {
            std::scoped_lock lock{mutex};
            stopped = true;
            for (auto& [_, entry] : entries) {
                entry->cancelled = true;
                entry->complete.notify_all();
            }
            for (auto& entry : waiting) {
                entry->cancelled = true;
                entry->complete.notify_all();
            }
            waiting.clear();
        }
        workers.reset();
        tasks.clear();
        entries.clear();
        // A later LoadDiskResources for another game in the same process must be able to admit
        // prebuilds again: upstream's flags stayed latched after the first Stop, so the second
        // and later game loads silently got zero deferred pipelines.
        captured_bytes = 0;
        started = false;
        stopped = false;
    }

private:
    EmissionBudget budget;
    const size_t capture_limit;
    size_t captured_bytes{};
    bool started{};
    bool stopped{};
    std::mutex mutex;
    std::unordered_map<Key, std::shared_ptr<Entry>> entries;
    std::vector<std::shared_ptr<Entry>> waiting;
    std::vector<Common::UniqueFunction<void>> tasks;
    std::unique_ptr<Common::ThreadWorker> workers;
};

} // namespace Vulkan
