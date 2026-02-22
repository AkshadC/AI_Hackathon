"use client";

import { useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import MainFeed from "@/components/MainFeed";
import RobotBuddy from "@/components/RobotBuddy";
import { type ThreadData } from "@/components/ThreadCard";

export default function Home() {
  const [queryThreads, setQueryThreads] = useState<ThreadData[]>([]);
  const [queryLoading, setQueryLoading] = useState(false);

  return (
    <main className="min-h-screen">
      <div className="flex min-h-screen">
        <MainFeed queryThreads={queryThreads} queryLoading={queryLoading} />
        <ChatPanel onQueryResult={setQueryThreads} onQueryLoading={setQueryLoading} />
        <RobotBuddy size={100} />
      </div>
    </main>
  );
}
