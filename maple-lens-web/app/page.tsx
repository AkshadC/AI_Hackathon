import ChatPanel from "@/components/ChatPanel";
import MainFeed from "@/components/MainFeed";

export default function Home() {
  return (
    <main className="min-h-screen">
      <div className="flex min-h-screen">
        <MainFeed />
        <ChatPanel />
      </div>
    </main>
  );
}