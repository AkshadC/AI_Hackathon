"use client";

import { useEffect, useState } from "react";

type TopicThread = {
  thread_id: string;
  title: string;
  url: string;
  score: number | null;
  num_comments: number | null;
};

type Topic = {
  topic_id: number;
  headline: string;
  summary: string;
  topic_trend_score: number;
  threads: TopicThread[];
};

export default function MainFeed() {
  const [topics, setTopics] = useState<Topic[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/topics")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setTopics(data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <section className="w-2/3 h-screen p-4 pt-23 flex flex-col gap-3">
      {/* Feed header */}
      <div className="mx-auto w-[90%]">
        <div className="text-lg font-semibold text-black pb-4">
          What&apos;s happening on r/Canada
        </div>
      </div>

      {/* Main feed container */}
      <div
        className="relative mx-auto w-[90%] h-[75vh] feed-scroll
          rounded-2xl overflow-hidden
          bg-gradient-to-br from-white/70 via-white/45 to-white/25
          backdrop-blur-2xl
          border border-white/40
          shadow-[0_20px_40px_rgba(0,0,0,0.10)]
          p-4 flex flex-col"
      >
        <div className="pointer-events-none absolute inset-x-0 top-0 h-24 rounded-2xl bg-gradient-to-b to-transparent" />

        <div className="relative flex-1 overflow-y-auto px-3 space-y-3">
          {loading && (
            <div className="text-center text-gray-500 py-8 animate-pulse">
              Loading topics...
            </div>
          )}

          {!loading && topics.length === 0 && (
            <div className="text-center text-gray-500 py-8">
              No trending topics yet. Run the summarization pipeline to generate topics.
            </div>
          )}

          {topics.map((topic) => (
            <div
              key={topic.topic_id}
              className="rounded-2xl bg-gradient-to-br from-white/80 to-white/55
                border border-white/40 shadow-sm p-4 space-y-2"
            >
              <div className="font-semibold text-gray-900">
                {topic.headline}
              </div>
              <div className="text-sm text-gray-700">
                {topic.summary}
              </div>
              <div className="text-xs text-gray-500">
                {topic.threads.length} thread{topic.threads.length !== 1 ? "s" : ""}
              </div>

              {/* Thread links */}
              {topic.threads.length > 0 && (
                <div className="space-y-1 pt-1">
                  {topic.threads.map((t) => (
                    <a
                      key={t.thread_id}
                      href={t.url || "#"}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-xs text-[#C45D1A] hover:underline truncate"
                      title={t.title}
                    >
                      {t.title}
                      {t.score != null && (
                        <span className="ml-2 text-gray-400">
                          {t.score} pts
                        </span>
                      )}
                    </a>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
