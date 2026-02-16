"use client";

import MiniChart from "./MiniChart";

type SentimentDist = { positive: number; neutral: number; negative: number };
type TimelinePoint = { date: string; count: number };

export type ThreadData = {
  thread_id: string;
  title: string;
  url: string;
  score: number;
  num_comments: number;
  similarity: number;
  sentiment_avg: number;
  sentiment_dist: SentimentDist;
  comment_timeline: TimelinePoint[];
};

function SentimentBadge({ avg }: { avg: number }) {
  let label: string;
  let color: string;

  if (avg >= 0.05) {
    label = "Positive";
    color = "bg-emerald-100 text-emerald-700";
  } else if (avg <= -0.05) {
    label = "Negative";
    color = "bg-red-100 text-red-700";
  } else {
    label = "Neutral";
    color = "bg-gray-100 text-gray-600";
  }

  return (
    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${color}`}>
      {label}
    </span>
  );
}

function SentimentBar({ dist }: { dist: SentimentDist }) {
  const total = dist.positive + dist.neutral + dist.negative;
  if (total === 0) return null;

  const pPct = (dist.positive / total) * 100;
  const nPct = (dist.neutral / total) * 100;

  return (
    <div className="flex h-1.5 w-full rounded-full overflow-hidden bg-gray-200">
      <div className="bg-emerald-400" style={{ width: `${pPct}%` }} />
      <div className="bg-gray-300" style={{ width: `${nPct}%` }} />
      <div className="bg-red-400 flex-1" />
    </div>
  );
}

export default function ThreadCard({ thread }: { thread: ThreadData }) {
  return (
    <div className="rounded-xl bg-gradient-to-br from-white/80 to-white/55 border border-white/40 shadow-sm p-3 space-y-2">
      {/* Title */}
      <a
        href={thread.url || "#"}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm font-semibold text-gray-900 hover:text-[#C45D1A] transition-colors line-clamp-2 block"
      >
        {thread.title}
      </a>

      {/* Stats row */}
      <div className="flex items-center gap-2 flex-wrap text-[10px] text-gray-500">
        <span title="Similarity">{Math.round(thread.similarity * 100)}% match</span>
        <span>|</span>
        <span title="Reddit score">{thread.score} pts</span>
        <span>|</span>
        <span title="Comments">{thread.num_comments} comments</span>
        <SentimentBadge avg={thread.sentiment_avg} />
      </div>

      {/* Sentiment bar */}
      <SentimentBar dist={thread.sentiment_dist} />

      {/* Mini chart */}
      {thread.comment_timeline.length > 1 && (
        <MiniChart data={thread.comment_timeline} />
      )}
    </div>
  );
}
