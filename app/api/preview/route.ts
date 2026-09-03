import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const inferenceUrl = process.env.SATQUERY_INFERENCE_URL ?? process.env.NEXT_PUBLIC_INFERENCE_URL;
  if (!inferenceUrl) return NextResponse.json({ error: 'Preview service is available with the local inference runtime.' }, { status: 503 });
  try {
    const form = await request.formData();
    const upstream = await fetch(`${inferenceUrl.replace(/\/$/, '')}/preview`, { method: 'POST', body: form, cache: 'no-store' });
    const payload = await upstream.json() as Record<string, unknown>;
    if (!upstream.ok) return NextResponse.json({ error: String(payload.detail ?? 'Preview service rejected the upload.') }, { status: upstream.status });
    return NextResponse.json(payload);
  } catch {
    return NextResponse.json({ error: 'Preview service is unavailable.' }, { status: 502 });
  }
}
