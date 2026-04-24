import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The Pi serves the Next.js app on port 3000 and the FastAPI on 8000.
  // We rewrite /api/* to the FastAPI so the browser uses the same origin.
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    return [
      { source: "/api/:path*", destination: `${apiBase}/api/:path*` },
      { source: "/static/:path*", destination: `${apiBase}/static/:path*` },
      { source: "/ws/:path*", destination: `${apiBase}/ws/:path*` },
    ];
  },
};

export default nextConfig;
