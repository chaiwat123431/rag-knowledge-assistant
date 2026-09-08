import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the workspace root to this folder. Without it, Turbopack walks up
  // and can latch onto a stray lockfile outside the repo.
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
