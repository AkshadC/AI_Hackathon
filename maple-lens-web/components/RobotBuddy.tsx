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
  defaultPos?: { x: number; y: number };
  size?: number;
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
  neutral: { label: "...", mouth: "flat" },
  happy: { label: "hiya!", mouth: "smile", blush: true },
  curious: { label: "hmm?", mouth: "tiny" },
  surprised: { label: "oh!", mouth: "o" },
  thinking: { label: "thinking...", mouth: "flat" },
  sleepy: { label: "zzz", mouth: "tiny" },
  excited: { label: "!", mouth: "zig", blush: true },
};

function pickRandom<T>(arr: T[]) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export default function RobotBuddy({
  defaultPos,
  size = 180,
  storageKey = "robotBuddyPos:v1",
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [pos, setPos] = useState<{ x: number; y: number }>(() => {
    return defaultPos ?? { x: 0, y: 0 };
  });

  // Pupil offset — circular tracking
  const [pupil, setPupil] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [emotion, setEmotion] = useState<Emotion>("neutral");
  const [bubble, setBubble] = useState<string>("...");
  const [showBubble, setShowBubble] = useState(false);

  // Drag state
  const dragRef = useRef<{
    dragging: boolean;
    pointerId: number | null;
    offsetX: number;
    offsetY: number;
  }>({ dragging: false, pointerId: null, offsetX: 0, offsetY: 0 });

  // Activity tracking
  const lastActivity = useRef<number>(now());
  const lastMoveReaction = useRef<number>(0);
  const typingDebounce = useRef<number | null>(null);
  const bubbleTimer = useRef<number | null>(null);

  const meta = EMOTION_META[emotion];

  const setMood = (m: Emotion, ttlMs = 1400) => {
    setEmotion(m);
    setBubble(EMOTION_META[m].label);
    setShowBubble(true);

    if (bubbleTimer.current) window.clearTimeout(bubbleTimer.current);
    bubbleTimer.current = window.setTimeout(() => {
      setEmotion((prev) => (prev === "sleepy" ? "sleepy" : "neutral"));
      setBubble("...");
      setShowBubble(false);
    }, ttlMs);
  };

  // Load saved position on mount
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

  // Global activity + idle + exclamation on move/type
  useEffect(() => {
    const onActivity = () => {
      lastActivity.current = now();
      setEmotion((prev) => (prev === "sleepy" ? "neutral" : prev));
    };

    const onMouseMove = () => {
      onActivity();
      // React with exclamation on fast cursor movement (throttled)
      const t = now();
      if (t - lastMoveReaction.current > 3000) {
        lastMoveReaction.current = t;
        setMood(pickRandom<Emotion>(["curious", "surprised", "excited"]), 1000);
      }
    };

    const onClick = () => {
      onActivity();
      setMood(pickRandom<Emotion>(["excited", "happy", "surprised"]), 900);
    };

    const onKeyDown = () => {
      onActivity();
      if (typingDebounce.current) window.clearTimeout(typingDebounce.current);
      typingDebounce.current = window.setTimeout(() => {
        setMood(pickRandom<Emotion>(["curious", "thinking", "excited", "surprised"]), 1200);
      }, 80);
    };

    window.addEventListener("pointerdown", onClick);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("scroll", onActivity, { passive: true });

    const idleTimer = window.setInterval(() => {
      const idleFor = now() - lastActivity.current;
      if (idleFor > 10000) {
        setEmotion("sleepy");
        setBubble("zzz");
        setShowBubble(true);
      }
    }, 1000);

    return () => {
      window.removeEventListener("pointerdown", onClick);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("mousemove", onMouseMove as any);
      window.removeEventListener("scroll", onActivity as any);
      window.clearInterval(idleTimer);
      if (typingDebounce.current) window.clearTimeout(typingDebounce.current);
      if (bubbleTimer.current) window.clearTimeout(bubbleTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cursor tracking — CIRCULAR pupil movement (not square)
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const el = containerRef.current;
      if (!el) return;

      const r = el.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;

      const dx = e.clientX - cx;
      const dy = e.clientY - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);

      const maxTravel = 10; // max pupil offset in px
      const sensitivity = 120; // distance at which pupil reaches max

      // Normalize to circular boundary
      const t = Math.min(dist / sensitivity, 1);
      const angle = Math.atan2(dy, dx);

      setPupil({
        x: Math.cos(angle) * t * maxTravel,
        y: Math.sin(angle) * t * maxTravel,
      });
    };

    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove as any);
  }, []);

  // Draggable handlers
  const onPointerDown = (e: React.PointerEvent) => {
    const el = containerRef.current;
    if (!el) return;

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

  // Eyelid for sleepy/thinking
  const eyeLid = useMemo(() => {
    if (emotion === "sleepy") return 12;
    if (emotion === "thinking") return 5;
    return 0;
  }, [emotion]);

  // Bouncy animation on excited/surprised
  const isAnimating = emotion === "excited" || emotion === "surprised";

  return (
    <div
      ref={containerRef}
      className="fixed z-[60] select-none"
      style={{
        left: pos.x,
        top: pos.y,
        width: size,
        height: size + 20, // extra space for speech bubble
        touchAction: "none",
        cursor: "grab",
      }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      role="button"
      aria-label="Robot buddy"
      title="Drag me!"
    >
      {/* Speech bubble */}
      <div
        className="absolute left-1/2 -translate-x-1/2 transition-all duration-300"
        style={{
          top: -36,
          opacity: showBubble ? 1 : 0,
          transform: `translateX(-50%) scale(${showBubble ? 1 : 0.7})`,
        }}
      >
        <div className="relative rounded-full bg-white/95 px-4 py-1.5 text-xs font-bold text-neutral-800 shadow-lg backdrop-blur whitespace-nowrap">
          {bubble}
          <div className="absolute left-1/2 top-full -translate-x-1/2 border-[6px] border-transparent border-t-white/95" />
        </div>
      </div>

      {/* Robot body — ROUND */}
      <div
        className="relative"
        style={{
          width: size,
          height: size,
          transition: "transform 0.2s ease",
          transform: isAnimating ? "scale(1.08)" : "scale(1)",
        }}
      >
        {/* Antenna */}
        <div className="absolute left-1/2 -translate-x-1/2" style={{ top: -2 }}>
          <div
            className="mx-auto rounded-full bg-neutral-300"
            style={{ width: 3, height: 18 }}
          />
          <div
            className="mx-auto -mt-1 rounded-full shadow-md ring-2 ring-white"
            style={{
              width: 12,
              height: 12,
              background: emotion === "excited" || emotion === "surprised"
                ? "#f87171"
                : emotion === "happy"
                ? "#fbbf24"
                : "#d4d4d4",
              transition: "background 0.3s ease",
            }}
          />
        </div>

        {/* Head — fully round */}
        <div
          className="absolute rounded-full bg-white shadow-xl ring-2 ring-neutral-100 overflow-hidden"
          style={{
            top: size * 0.1,
            left: size * 0.08,
            width: size * 0.84,
            height: size * 0.84,
          }}
        >
          {/* Top gloss */}
          <div
            className="absolute rounded-full bg-gradient-to-b from-white/80 to-transparent"
            style={{
              top: 4,
              left: "15%",
              width: "70%",
              height: "30%",
            }}
          />

          {/* Eyes container */}
          <div
            className="absolute left-0 right-0 flex items-center justify-center"
            style={{ top: "30%", gap: size * 0.2 }}
          >
            {/* Left eye */}
            <div
              className="relative rounded-full bg-neutral-100"
              style={{ width: size * 0.2, height: size * 0.2 }}
            >
              {/* Pupil */}
              <div
                className="absolute rounded-full bg-neutral-900"
                style={{
                  width: size * 0.1,
                  height: size * 0.1,
                  left: "50%",
                  top: "50%",
                  transform: `translate(-50%, -50%) translate(${pupil.x}px, ${pupil.y}px)`,
                  transition: "transform 0.08s ease-out",
                }}
              />
              {/* Pupil shine */}
              <div
                className="absolute rounded-full bg-white"
                style={{
                  width: size * 0.035,
                  height: size * 0.035,
                  left: `calc(50% + ${pupil.x * 0.6 + 3}px)`,
                  top: `calc(50% + ${pupil.y * 0.6 - 3}px)`,
                  transform: "translate(-50%, -50%)",
                  transition: "left 0.08s ease-out, top 0.08s ease-out",
                }}
              />
              {/* Eyelid (sleepy/thinking) */}
              {eyeLid > 0 && (
                <div
                  className="absolute top-0 left-0 right-0 rounded-t-full bg-white z-10"
                  style={{
                    height: `${eyeLid * (100 / (size * 0.2))}%`,
                    maxHeight: "70%",
                    transition: "height 0.3s ease",
                  }}
                />
              )}
            </div>

            {/* Right eye */}
            <div
              className="relative rounded-full bg-neutral-100"
              style={{ width: size * 0.2, height: size * 0.2 }}
            >
              <div
                className="absolute rounded-full bg-neutral-900"
                style={{
                  width: size * 0.1,
                  height: size * 0.1,
                  left: "50%",
                  top: "50%",
                  transform: `translate(-50%, -50%) translate(${pupil.x}px, ${pupil.y}px)`,
                  transition: "transform 0.08s ease-out",
                }}
              />
              <div
                className="absolute rounded-full bg-white"
                style={{
                  width: size * 0.035,
                  height: size * 0.035,
                  left: `calc(50% + ${pupil.x * 0.6 + 3}px)`,
                  top: `calc(50% + ${pupil.y * 0.6 - 3}px)`,
                  transform: "translate(-50%, -50%)",
                  transition: "left 0.08s ease-out, top 0.08s ease-out",
                }}
              />
              {eyeLid > 0 && (
                <div
                  className="absolute top-0 left-0 right-0 rounded-t-full bg-white z-10"
                  style={{
                    height: `${eyeLid * (100 / (size * 0.2))}%`,
                    maxHeight: "70%",
                    transition: "height 0.3s ease",
                  }}
                />
              )}
            </div>
          </div>

          {/* Blush */}
          {meta.blush && (
            <>
              <div
                className="absolute rounded-full bg-rose-300/50 blur-[2px]"
                style={{
                  width: size * 0.14,
                  height: size * 0.06,
                  left: "12%",
                  top: "58%",
                  transition: "opacity 0.3s ease",
                }}
              />
              <div
                className="absolute rounded-full bg-rose-300/50 blur-[2px]"
                style={{
                  width: size * 0.14,
                  height: size * 0.06,
                  right: "12%",
                  top: "58%",
                  transition: "opacity 0.3s ease",
                }}
              />
            </>
          )}

          {/* Mouth */}
          <div
            className="absolute left-0 right-0 flex justify-center"
            style={{ top: "66%" }}
          >
            {meta.mouth === "flat" && (
              <div
                className="rounded-full bg-neutral-700"
                style={{ width: size * 0.16, height: 3 }}
              />
            )}
            {meta.mouth === "smile" && (
              <svg
                width={size * 0.22}
                height={size * 0.12}
                viewBox="0 0 40 20"
              >
                <path
                  d="M4 6 Q20 22 36 6"
                  stroke="#404040"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                />
              </svg>
            )}
            {meta.mouth === "o" && (
              <div
                className="rounded-full border-[3px] border-neutral-700"
                style={{ width: size * 0.1, height: size * 0.1 }}
              />
            )}
            {meta.mouth === "zig" && (
              <svg
                width={size * 0.26}
                height={size * 0.1}
                viewBox="0 0 44 14"
              >
                <path
                  d="M2 7 L10 2 L18 12 L26 2 L34 12 L42 7"
                  stroke="#404040"
                  strokeWidth="3"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            )}
            {meta.mouth === "tiny" && (
              <div
                className="rounded-full bg-neutral-600"
                style={{ width: size * 0.1, height: 2.5 }}
              />
            )}
          </div>
        </div>

        {/* Shadow underneath */}
        <div
          className="absolute rounded-full bg-black/10 blur-md"
          style={{
            bottom: 0,
            left: "20%",
            width: "60%",
            height: 8,
          }}
        />
      </div>
    </div>
  );
}
