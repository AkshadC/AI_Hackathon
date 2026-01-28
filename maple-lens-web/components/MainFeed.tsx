export default function MainFeed() {
  return (
    <section className="w-2/3 h-screen  p-4 pt-23 flex flex-col gap-3">



      {/* Feed header aligned with Maple Bot text (same vibe) */}
      <div className="mx-auto w-[90%]">
        <div className="text-lg font-semibold text-black pb-4">What’s happening on r/Canada</div>
      </div>

      {/* Main feed container (glass like chat) */}
      <div
          className="relative mx-auto w-[90%] h-[75vh] feed-scroll
          rounded-2xl overflow-hidden
          bg-gradient-to-br from-white/60 via-white/35 to-white/20
          backdrop-blur-2xl
          border border-white/40
          shadow-[0_20px_40px_rgba(0,0,0,0.10)]
          p-4 flex flex-col "
        >

          {/* Gloss highlight */}
          <div className="pointer-events-none absolute inset-x-0 top-0 h-24 rounded-2xl bg-gradient-to-b to-transparent" />

          {/* Feed items */}
          <div className="relative flex-1 overflow-y-auto px-3 space-y-3 ">
            {Array.from({ length: 30 }).map((_, i) => (
              <div
                key={i}
                className="rounded-2xl bg-gradient-to-br from-white/80 to-white/55
                border border-white/40 shadow-sm p-4"
              >
                <div className="font-semibold text-gray-900">
                  Feed item #{i + 1}
                </div>
                <div className="text-sm text-gray-700 mt-1">
                  Feed content here
                </div>

              </div>
            ))}
          </div>
        </div>
    </section>
  );
}
