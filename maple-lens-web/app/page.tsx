import ChatPanel from "@/components/ChatPanel";
import MainFeed from "@/components/MainFeed";
import RobotBuddy from "@/components/RobotBuddy";
export default function Home() {
  return (
    <main className="min-h-screen">
      <div className="flex min-h-screen">
        <MainFeed />
        <ChatPanel />
          <RobotBuddy size={100} />
      </div>

    </main>
  );
}