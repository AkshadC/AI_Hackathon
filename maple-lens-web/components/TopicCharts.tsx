"use client";

import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

type SentimentDist = { positive: number; neutral: number; negative: number };
type TimelinePoint = { date: string; count: number };

type Topic = {
  topic_id: number;
  headline: string;
  summary: string;
  topic_trend_score: number;
  sentiment_dist: SentimentDist;
  comment_timeline: TimelinePoint[];
  total_comments: number;
  last_updated: string | null;
  threads: unknown[];
};

const SENTIMENT_COLORS = ["#10b981", "#9ca3af", "#ef4444"]; // green, gray, red

export default function TopicCharts({
  topics,
  overallTimeline,
}: {
  topics: Topic[];
  overallTimeline: TimelinePoint[];
}) {
  if (!topics || topics.length === 0) return null;

  // --- 1. Sentiment donut: aggregate across all topics ---
  const aggSentiment = topics.reduce(
    (acc, t) => {
      const sd = t.sentiment_dist || { positive: 0, neutral: 0, negative: 0 };
      acc.positive += sd.positive;
      acc.neutral += sd.neutral;
      acc.negative += sd.negative;
      return acc;
    },
    { positive: 0, neutral: 0, negative: 0 }
  );
  const sentimentData = [
    { name: "Positive", value: aggSentiment.positive },
    { name: "Neutral", value: aggSentiment.neutral },
    { name: "Negative", value: aggSentiment.negative },
  ].filter((d) => d.value > 0);

  // --- 2. Topic trend bars ---
  const trendData = topics
    .map((t) => ({
      name: t.headline.length > 18 ? t.headline.slice(0, 18) + "..." : t.headline,
      score: Math.round(t.topic_trend_score * 100) / 100,
    }))
    .sort((a, b) => b.score - a.score);

  // --- 3. Sentiment over time: compute daily avg sentiment from topic timelines ---
  // We approximate using each topic's overall sentiment weighted by daily volume
  // For simplicity, we use the overall timeline dates and show the aggregate sentiment ratio
  const sentimentOverTime = overallTimeline.slice(-14).map((point) => {
    // Find which topics had activity on this date and aggregate sentiment
    let pos = 0, neu = 0, neg = 0;
    for (const t of topics) {
      const dayEntry = (t.comment_timeline || []).find((c) => c.date === point.date);
      if (dayEntry && t.total_comments > 0) {
        const sd = t.sentiment_dist || { positive: 0, neutral: 0, negative: 0 };
        const total = sd.positive + sd.neutral + sd.negative;
        if (total > 0) {
          const weight = dayEntry.count;
          pos += (sd.positive / total) * weight;
          neg += (sd.negative / total) * weight;
          neu += (sd.neutral / total) * weight;
        }
      }
    }
    const total = pos + neu + neg;
    return {
      date: point.date.slice(5), // "MM-DD"
      sentiment: total > 0 ? Math.round(((pos - neg) / total) * 100) / 100 : 0,
    };
  });

  const chartCard =
    "rounded-2xl bg-gradient-to-br from-white/80 to-white/55 border border-white/40 shadow-sm p-4";

  return (
    <div className="grid grid-cols-2 gap-3">
      {/* Comment Activity */}
      <div className={chartCard}>
        <div className="text-xs font-semibold text-gray-700 mb-2">Comment Activity (14d)</div>
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={overallTimeline.slice(-14).map((p) => ({ ...p, date: p.date.slice(5) }))}>
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} width={30} />
            <Tooltip />
            <Area
              type="monotone"
              dataKey="count"
              stroke="#C45D1A"
              fill="#C45D1A"
              fillOpacity={0.2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Sentiment Donut */}
      <div className={chartCard}>
        <div className="text-xs font-semibold text-gray-700 mb-2">Overall Sentiment</div>
        <ResponsiveContainer width="100%" height={160}>
          <PieChart>
            <Pie
              data={sentimentData}
              cx="50%"
              cy="50%"
              innerRadius={35}
              outerRadius={60}
              paddingAngle={3}
              dataKey="value"
            >
              {sentimentData.map((_, i) => (
                <Cell key={i} fill={SENTIMENT_COLORS[["Positive", "Neutral", "Negative"].indexOf(sentimentData[i].name)]} />
              ))}
            </Pie>
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Topic Trend Bars */}
      <div className={chartCard}>
        <div className="text-xs font-semibold text-gray-700 mb-2">Topic Trend Scores</div>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={trendData} layout="vertical">
            <XAxis type="number" tick={{ fontSize: 10 }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 9 }} width={80} />
            <Tooltip />
            <Bar dataKey="score" fill="#C45D1A" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Sentiment Over Time */}
      <div className={chartCard}>
        <div className="text-xs font-semibold text-gray-700 mb-2">Sentiment Over Time</div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={sentimentOverTime}>
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} width={30} domain={[-1, 1]} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="sentiment"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
