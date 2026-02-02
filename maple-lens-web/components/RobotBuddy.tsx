"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

type Emotion =
  | "neutral"
  | "happy"
  | "curious"
  | "surprised"
  | "thinking"
  | "sleepy"
  | "excited";

type Props = {
  /** If you want a different default start position */
  defaultPos?: { x: number; y: number };
  /** Size in px */
  size?: number;
  /** Optional: give a key to persist multiple robots separately */
  storageKey?: string;
};

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

function now() {
  return Date.now();
}

const EMOTION_META: Record<
  Emotion,
  { label: string; mouth: "smile" | "flat" | "o" | "zig" | "tiny"; blush?: boolean }
> = {
  neutral: { label: "…", mouth: "flat" },
  happy: { label: "hiya!", mouth: "smile", blush: true },
  curious: { label: "hmm?", mouth: "tiny" },
  surprised: { label: "oh!", mouth: "o" },
  thinking: { label: "thinking…", mouth: "flat" },
  sleepy: { label: "zzz", mouth: "tiny" },
  excited: { label: "!", mouth: "zig", blush: true },
};

function pickRandom<T>(arr: T[]) {
  return arr[Math.floor(Math.random() * arr.length)];
}

/**
 * RobotBuddy
 * - Cursor tracking pupils
 * - Emotion state machine based on events + idle timer
 * - Draggable; persists position in localStorage
 */
export default function RobotBuddy({
  defaultPos,
  size = 140,
  storageKey = "robotBuddyPos:v1",
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Position (draggable)
  const [pos, setPos] = useState<{ x: number; y: number }>(() => {
    // Start in middle of screen by default (client will correct after mount)
    return defaultPos ?? { x: 0, y: 0 };
  });

  // Cursor tracking (pupil offsets)
  const [pupil, setPupil] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Emotions
  const [emotion, setEmotion] = useState<Emotion>("neutral");
  const [bubble, setBubble] = useState<string>("…");

  // Drag state
  const dragRef = useRef<{
    dragging: boolean;
    pointerId: number | null;
    offsetX: number;
    offsetY: number;
  }>({ dragging: false, pointerId: null, offsetX: 0, offsetY: 0 });

  // Activity tracking (for idle/sleepy)
  const lastActivity = useRef<number>(now());
  const typingDebounce = useRef<number | null>(null);
  const bubbleTimer = useRef<number | null>(null);

  const meta = EMOTION_META[emotion];

  // Helper: set emotion with a “speech bubble” and optional auto-return
  const setMood = (m: Emotion, ttlMs = 1400) => {
    setEmotion(m);
    setBubble(EMOTION_META[m].label);

    if (bubbleTimer.current) window.clearTimeout(bubbleTimer.current);
    bubbleTimer.current = window.setTimeout(() => {
      // Fade back to neutral unless user became idle (handled elsewhere)
      setEmotion((prev) => (prev === "sleepy" ? "sleepy" : "neutral"));
      setBubble("…");
    }, ttlMs);
  };

  // Load saved position + center default on mount
  useEffect(() => {
    const readSaved = () => {
      try {
        const raw = localStorage.getItem(storageKey);
        if (raw) return JSON.parse(raw) as { x: number; y: number };
      } catch {}
      return null;
    };

    const saved = readSaved();
    const w = window.innerWidth;
    const h = window.innerHeight;

    // If saved exists, use it; otherwise center
    const start =
      saved ??
      defaultPos ?? {
        x: Math.round(w / 2 - size / 2),
        y: Math.round(h / 2 - size / 2),
      };

    setPos({
      x: clamp(start.x, 8, Math.max(8, w - size - 8)),
      y: clamp(start.y, 8, Math.max(8, h - size - 8)),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist position
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(pos));
    } catch {}
  }, [pos, storageKey]);

  // Global activity + idle handling
  useEffect(() => {
    const onActivity = () => {
      lastActivity.current = now();
      // If waking from sleepy
      setEmotion((prev) => (prev === "sleepy" ? "neutral" : prev));
    };

    const onClick = () => {
      onActivity();
      setMood(pickRandom<Emotion>(["excited", "happy", "surprised"]), 900);
    };

    const onKeyDown = () => {
      onActivity();
      // “curious when user types something”
      if (typingDebounce.current) window.clearTimeout(typingDebounce.current);
      typingDebounce.current = window.setTimeout(() => {
        setMood(pickRandom<Emotion>(["curious", "thinking"]), 1200);
      }, 80);
    };

    window.addEventListener("pointerdown", onClick);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("mousemove", onActivity, { passive: true });
    window.addEventListener("scroll", onActivity, { passive: true });

    const idleTimer = window.setInterval(() => {
      const idleFor = now() - lastActivity.current;
      // after 10s idle -> sleepy
      if (idleFor > 10000) {
        setEmotion("sleepy");
        setBubble("zzz");
      }
    }, 1000);

    return () => {
      window.removeEventListener("pointerdown", onClick);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("mousemove", onActivity as any);
      window.removeEventListener("scroll", onActivity as any);
      window.clearInterval(idleTimer);
      if (typingDebounce.current) window.clearTimeout(typingDebounce.current);
      if (bubbleTimer.current) window.clearTimeout(bubbleTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cursor tracking: compute pupil offset relative to robot center
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const el = containerRef.current;
      if (!el) return;

      const r = el.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;

      const dx = e.clientX - cx;
      const dy = e.clientY - cy;

      // Distance-based scaling (feels way more natural)
      const max = 8;     // max pupil travel in px (increase if you want)
      const scale = 80;   // smaller = reacts more aggressively

      const nx = clamp((dx / scale) * max, -max, max);
      const ny = clamp((dy / scale) * max, -max, max);

      setPupil({ x: nx, y: ny });


      setPupil({ x: clamp(nx, -max, max), y: clamp(ny, -max, max) });

      // Small chance to look “curious” when hovering around it
      if (r.left <= e.clientX && e.clientX <= r.right && r.top <= e.clientY && e.clientY <= r.bottom) {
        if (Math.random() < 0.01) setMood("curious", 900);
      }
    };

    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove as any);
  }, []);

  // Draggable handlers
  const onPointerDown = (e: React.PointerEvent) => {
    const el = containerRef.current;
    if (!el) return;

    // Prevent dragging from selecting text / causing weirdness
    e.preventDefault();
    (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);

    dragRef.current.dragging = true;
    dragRef.current.pointerId = e.pointerId;

    const rect = el.getBoundingClientRect();
    dragRef.current.offsetX = e.clientX - rect.left;
    dragRef.current.offsetY = e.clientY - rect.top;

    lastActivity.current = now();
    setMood("excited", 700);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragRef.current.dragging) return;

    const w = window.innerWidth;
    const h = window.innerHeight;

    const x = e.clientX - dragRef.current.offsetX;
    const y = e.clientY - dragRef.current.offsetY;

    setPos({
      x: clamp(x, 8, Math.max(8, w - size - 8)),
      y: clamp(y, 8, Math.max(8, h - size - 8)),
    });
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (dragRef.current.pointerId !== e.pointerId) return;
    dragRef.current.dragging = false;
    dragRef.current.pointerId = null;
    lastActivity.current = now();
    setMood("happy", 900);
  };

  // Visual tweaks based on emotion
  const eyeLid = useMemo(() => {
    if (emotion === "sleepy") return 10; // eyelid height
    if (emotion === "thinking") return 4;
    return 0;
  }, [emotion]);

  return (
    <div
      ref={containerRef}
      className="fixed z-[60] select-none"
      style={{
        left: pos.x,
        top: pos.y,
        width: size,
        height: size,
        touchAction: "none",
      }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      role="button"
      aria-label="Robot buddy"
      title="Drag me!"
    >
      {/* speech bubble */}
      <div className="absolute -top-10 left-1/2 -translate-x-1/2">
        <div className="relative rounded-full bg-white/90 px-3 py-1 text-xs font-semibold text-neutral-800 shadow-md backdrop-blur">
          {bubble}
          <div className="absolute left-1/2 top-full -translate-x-1/2 border-8 border-transparent border-t-white/90" />
        </div>
      </div>

      {/* body */}
      <div className="relative h-full w-full">
        {/* antenna */}
        <div className="absolute left-1/2 top-2 -translate-x-1/2">
          <div className="h-6 w-1 rounded-full bg-neutral-200/90 shadow-sm" />
          <div className="mx-auto -mt-1 h-3 w-3 rounded-full bg-white shadow-md ring-2 ring-neutral-200" />
        </div>

        {/* head */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative h-[78%] w-[78%] rounded-[28px] bg-white shadow-lg ring-1 ring-neutral-200">
            {/* top highlight */}
            <div className="absolute left-3 right-3 top-3 h-3 rounded-full bg-neutral-100" />

          {/* eyes (dots only) */}
          <div className="absolute left-0 right-0 top-[38%] flex items-center justify-center gap-10">
            {/* left dot */}
            <div className="relative h-3 w-3">
              <div
                className="absolute left-1/2 top-1/2 h-3 w-3 rounded-full bg-neutral-900"
                style={{
                  transform: `translate(-50%, -50%) translate(${pupil.x}px, ${pupil.y}px)`,
                }}
              />
            </div>

            {/* right dot */}
            <div className="relative h-3 w-3">
              <div
                className="absolute left-1/2 top-1/2 h-3 w-3 rounded-full bg-neutral-900"
                style={{
                  transform: `translate(-50%, -50%) translate(${pupil.x}px, ${pupil.y}px)`,
                }}
              />
            </div>
          </div>

            {/* blush */}
            {meta.blush && (
              <>
                <div className="absolute left-5 top-[54%] h-3 w-6 rounded-full bg-rose-200/70 blur-[0.2px]" />
                <div className="absolute right-5 top-[54%] h-3 w-6 rounded-full bg-rose-200/70 blur-[0.2px]" />
              </>
            )}

            {/* mouth */}
            <div className="absolute left-0 right-0 top-[62%] flex justify-center">
              {meta.mouth === "flat" && (
                <div className="h-1 w-10 rounded-full bg-neutral-800/80" />
              )}
              {meta.mouth === "smile" && (
                <div className="h-6 w-12 rounded-b-full border-b-4 border-neutral-800/80" />
              )}
              {meta.mouth === "o" && (
                <div className="h-6 w-6 rounded-full border-4 border-neutral-800/80" />
              )}
              {meta.mouth === "zig" && (
                <svg width="44" height="14" viewBox="0 0 44 14" className="opacity-90">
                  <path
                    d="M2 7 L10 2 L18 12 L26 2 L34 12 L42 7"
                    stroke="currentColor"
                    strokeWidth="3"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-neutral-800"
                  />
                </svg>
              )}
              {meta.mouth === "tiny" && (
                <div className="h-1 w-6 rounded-full bg-neutral-800/70" />
              )}
            </div>
          </div>
        </div>

        {/* subtle shadow base */}
        <div className="absolute bottom-2 left-1/2 h-3 w-[70%] -translate-x-1/2 rounded-full bg-black/10 blur-sm" />
      </div>
    </div>
  );
}
