import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  experimental: {
    // The upload route forwards compact GeoTIFF pairs to the local inference
    // service. Keep this above the bundled test-pair size (about 18 MB).
    serverActions: { bodySizeLimit: '50mb' },
  },
};

export default nextConfig;
