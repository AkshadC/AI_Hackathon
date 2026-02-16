import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {/* Header */}
        <header className="w-full px-6 py-4 flex items-center justify-between border-b border-white/10 bg-gradient-to-r from-[#1a1a2e] to-[#16213e]">
          <div className="text-lg font-semibold text-white">🍁 Maple Lens</div>
          <nav className="text-sm text-white">
            Reddit AI Search
          </nav>
        </header>

        {/* Page content */}
        <main>{children}</main>

        {/* Footer */}
        <footer className="w-full px-6 py-4 text-center text-sm text-white/70">
          © {new Date().getFullYear()} Maple Lens · Built for AI Hackathon
        </footer>
      </body>
    </html>
  );
}
