import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'SatQuery AI — SENTRY',
  description: 'Sensor-aware evidence for remote-sensing disaster intelligence.',
  openGraph: {
    title: 'SatQuery AI — SENTRY',
    description: 'Sensor-aware evidence for remote-sensing disaster intelligence.',
    type: 'website',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'SatQuery AI — Earth, under evidence.' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'SatQuery AI — SENTRY',
    description: 'Sensor-aware evidence for remote-sensing disaster intelligence.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
