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
        <header className="w-full px-6 py-4 flex items-center justify-between border-black border-b-2 bg-black">
          <div className="text-lg font-semibold text-white">🍁 Maple Lens</div>
          <nav className="text-sm text-white">
            Reddit AI Search
          </nav>
        </header>

        {/* Page content */}
        <main>{children}</main>

        {/* Footer */}
        <footer className="w-full px-6 py-4 text-center text-sm text-black/60">
          © {new Date().getFullYear()} Maple Lens · Built for AI Hackathon
        </footer>
      </body>
    </html>
  );
}
