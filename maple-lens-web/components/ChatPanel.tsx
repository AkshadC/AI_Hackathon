"use client";

import Image from "next/image";
import { useState, useRef, useEffect } from "react";

export default function ChatPanel() {
    const bottomRef = useRef<HTMLDivElement | null>(null);
    const [msg, setMsg] = useState("");
    const [chat, setChat] = useState([
        {role: "bot", text: "Hi! Ask me anything about r/Canada topics."},
    ]);

    async function send() {
        const t = msg.trim();
        if (!t) return;
        setChat((c) => [...c, {role: "user", text: t}]);
        setMsg("");
        const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: t }),
          });

        const data = await res.json();

        setChat((c) => [...c, { role: "bot", text: data.reply }]);

    }
    useEffect(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chat]);

    return (
  <aside className="w-1/3 h-screen p-4 flex flex-col gap-3">

    {/* Header OUTSIDE the chat container */}
    <div className="relative mx-auto h-14 w-14 rounded-full bg-gradient-to-br from-[#DC6602] to-[#FCA505] flex items-center justify-center text-white text-2xl shadow-lg">
      <div className="absolute top-1 left-1 h-5 w-10 rounded-full bg-white/40 blur-sm" />
      🍁
    </div>

      <div className="flex flex-col items-center gap-1">
        <div className="font-semibold text-black text-lg">
          Maple Bot
        </div>
        <div className="text-sm text-black">
          r/Canada insights
        </div>
      </div>


    {/* Chat container */}
    <div className="relative h-[75vh] w-[45vh] mx-auto rounded-2xl
      bg-gradient-to-br from-white/60 via-white/35 to-white/20
      backdrop-blur-2xl
      border border-white/40
      shadow-[0_20px_40px_rgba(0,0,0,0.10)]
      flex flex-col">
    <div className="pointer-events-none absolute inset-x-0 top-0 h-24 rounded-2xl bg-gradient-to-b from-white/20 to-transparent" />

      {/* Messages */}
      <div className="chat-scroll mt-4 flex-1 overflow-y-auto space-y-3 px-4 pr-3">
        {chat.map((m, i) => (
          <div
            key={i}
            className={`max-w-[85%] rounded-2xl px-4 py-2 text-sm shadow-sm ${
              m.role === "user"
                ? "ml-auto mr-2 bg-gradient-to-br from-[#2f855a] to-[#276749] text-white"
                : "mr-auto ml-2 bg-gradient-to-br from-white/80 to-white/55 text-gray-900"
            }`}
          >
            {m.text}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 flex gap-2 border-t border-[#51561F]/30">
        <input
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Type your question…"
          className="flex-1 rounded-2xl border border-[#51561F]/40 bg-white/70 px-4 py-2 outline-none"
        />
        <button
          onClick={send}
          className="rounded-2xl bg-black px-4 py-2 text-white hover:bg-[#2B4F81]/75 "
        >
          Send
        </button>
      </div>

    </div>
  </aside>
);

}
