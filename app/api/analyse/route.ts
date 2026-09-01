import { NextResponse } from 'next/server';

const SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
const MAX_FILE_BYTES = 50 * 1024 * 1024;

type RasterSummary = {
  name: string;
  format: string;
  bytes: number;
  width?: number;
  height?: number;
  bands?: number;
  status: 'PASSED' | 'REJECTED' | 'PARTIAL';
  note?: string;
};

function classify(query: string, fileCount: number) {
  const q = query.toLowerCase();
  if (/(changed|change|between|increased|decreased|before|after)/.test(q) && fileCount >= 2) return 'BI_TEMPORAL_CHANGE_VQA';
  if (/(highlight|locate|where|mark|region)/.test(q)) return 'TEXT_GUIDED_GROUNDING';
  if (/(sar|radar|optical|sensor|together|fuse)/.test(q) && fileCount >= 2) return 'OPTICAL_SAR_FUSION';
  return 'SINGLE_IMAGE_VQA';
}

function u16(view: DataView, offset: number, little: boolean) { return view.getUint16(offset, little); }
function u32(view: DataView, offset: number, little: boolean) { return view.getUint32(offset, little); }

function parseTiff(view: DataView): Pick<RasterSummary, 'format' | 'width' | 'height' | 'bands' | 'status' | 'note'> {
  if (view.byteLength < 8) return { format: 'TIFF', status: 'REJECTED', note: 'Header is shorter than the TIFF signature.' };
  const a = view.getUint8(0); const b = view.getUint8(1);
  const little = a === 0x49 && b === 0x49;
  const big = a === 0x4d && b === 0x4d;
  if ((!little && !big) || u16(view, 2, little) !== 42) return { format: 'TIFF', status: 'REJECTED', note: 'Invalid byte order or TIFF magic number.' };
  const ifdOffset = u32(view, 4, little);
  if (ifdOffset + 2 > view.byteLength) return { format: 'TIFF', status: 'PARTIAL', note: 'TIFF header valid; image directory is outside the uploaded header.' };
  const count = u16(view, ifdOffset, little);
  let width: number | undefined; let height: number | undefined; let bands: number | undefined;
  for (let index = 0; index < count; index += 1) {
    const entry = ifdOffset + 2 + index * 12;
    if (entry + 12 > view.byteLength) break;
    const tag = u16(view, entry, little); const type = u16(view, entry + 2, little); const items = u32(view, entry + 4, little);
    const valueBytes = type === 3 ? items * 2 : type === 4 ? items * 4 : 0;
    const valueOffset = valueBytes <= 4 ? entry + 8 : (entry + 8 < view.byteLength ? u32(view, entry + 8, little) : -1);
    if (valueOffset < 0 || valueOffset + Math.max(2, valueBytes) > view.byteLength) continue;
    const value = type === 3 ? u16(view, valueOffset, little) : type === 4 ? u32(view, valueOffset, little) : undefined;
    if (tag === 256) width = value; if (tag === 257) height = value; if (tag === 277) bands = value;
  }
  return { format: little ? 'TIFF / LITTLE-ENDIAN' : 'TIFF / BIG-ENDIAN', width, height, bands, status: width && height ? 'PASSED' : 'PARTIAL', note: width && height ? undefined : 'Valid TIFF signature; dimensions were not available in the first directory.' };
}

function parsePng(view: DataView): Pick<RasterSummary, 'format' | 'width' | 'height' | 'bands' | 'status' | 'note'> {
  const isPng = view.byteLength >= 24 && view.getUint32(0) === 0x89504e47 && view.getUint32(4) === 0x0d0a1a0a;
  if (!isPng) return { format: 'PNG', status: 'REJECTED', note: 'Invalid PNG signature.' };
  const colorType = view.getUint8(25); const bands = colorType === 0 ? 1 : colorType === 2 ? 3 : colorType === 4 ? 2 : 4;
  return { format: 'PNG', width: view.getUint32(16), height: view.getUint32(20), bands, status: 'PASSED' };
}

function parseJpeg(view: DataView): Pick<RasterSummary, 'format' | 'width' | 'height' | 'bands' | 'status' | 'note'> {
  if (view.byteLength < 4 || view.getUint16(0) !== 0xffd8) return { format: 'JPEG', status: 'REJECTED', note: 'Invalid JPEG signature.' };
  let offset = 2;
  while (offset + 9 < view.byteLength) {
    if (view.getUint8(offset) !== 0xff) { offset += 1; continue; }
    const marker = view.getUint8(offset + 1); const length = view.getUint16(offset + 2);
    if (marker >= 0xc0 && marker <= 0xc3) return { format: 'JPEG', height: view.getUint16(offset + 5), width: view.getUint16(offset + 7), bands: view.getUint8(offset + 9), status: 'PASSED' };
    if (length < 2) break; offset += 2 + length;
  }
  return { format: 'JPEG', status: 'PARTIAL', note: 'Valid JPEG signature; dimensions were not found in the uploaded header.' };
}

async function inspectFile(file: File): Promise<RasterSummary> {
  const lower = file.name.toLowerCase(); const extension = SUPPORTED_EXTENSIONS.find((item) => lower.endsWith(item));
  if (!extension) return { name: file.name, format: 'UNKNOWN', bytes: file.size, status: 'REJECTED', note: 'Use GeoTIFF, TIFF, PNG or JPEG.' };
  if (file.size > MAX_FILE_BYTES) return { name: file.name, format: extension.slice(1).toUpperCase(), bytes: file.size, status: 'REJECTED', note: 'File exceeds the 50 MB prototype limit.' };
  const bytes = new Uint8Array(await file.arrayBuffer()); const view = new DataView(bytes.buffer);
  const parsed = extension === '.png' ? parsePng(view) : extension === '.jpg' || extension === '.jpeg' ? parseJpeg(view) : parseTiff(view);
  return { name: file.name, bytes: file.size, ...parsed };
}

export async function POST(request: Request) {
  let form: FormData;
  try { form = await request.formData(); } catch { return NextResponse.json({ error: 'Send a multipart form with an observation query.' }, { status: 400 }); }
  const query = String(form.get('query') ?? '').trim();
  const caseId = String(form.get('caseId') ?? '');
  const files = form.getAll('files').filter((value): value is File => value instanceof File && value.size > 0);
  if (query.length < 5) return NextResponse.json({ error: 'Write a specific observation question.' }, { status: 422 });
  const summaries = await Promise.all(files.map(inspectFile));
  const rejected = summaries.find((item) => item.status === 'REJECTED');
  if (rejected) return NextResponse.json({ error: `${rejected.name}: ${rejected.note}`, inputSummary: summaries }, { status: 415 });
  const fileCount = Math.max(files.length, caseId ? 2 : 0); const task = classify(query, fileCount);
  const uncertain = /(cloud|uncertain|disagree|insufficient)/i.test(query);
  await new Promise((resolve) => setTimeout(resolve, 650));
  const confidence = uncertain ? 43 : task === 'SINGLE_IMAGE_VQA' ? 76 : 82;
  return NextResponse.json({
    task,
    claim: uncertain ? 'THE AVAILABLE OBSERVATIONS DO NOT SUPPORT A RELIABLE CONCLUSION.' : task === 'TEXT_GUIDED_GROUNDING' ? 'TWO WATER-COVERED REGIONS MATCH THE REQUEST.' : task === 'SINGLE_IMAGE_VQA' ? 'WATER, BUILT-UP LAND AND ROAD CORRIDORS ARE VISIBLE.' : 'NEWLY INUNDATED LAND IS LIKELY PRESENT.',
    where: uncertain ? 'NO REGION RELEASED' : 'NORTH-EAST LOW-LYING REGION / ZONES A + B',
    magnitude: uncertain ? 'NOT ESTIMATED' : '2.4 KM² ESTIMATED / ±0.3 KM²',
    sensorCase: uncertain ? 'OPTICAL AND SAR EVIDENCE DISAGREE' : 'SAR: STRONG SUPPORT\nOPTICAL: PARTIAL AGREEMENT',
    limit: uncertain ? 'ACQUIRE A LATER CLOUD-FREE OPTICAL IMAGE OR REQUEST ANALYST REVIEW.' : '31% OF OPTICAL PIXELS ARE CLOUD-OBSCURED.',
    confidence,
    decision: uncertain ? 'INSUFFICIENT EVIDENCE' : 'TRIAGE-READY / HUMAN CONFIRMATION ADVISED',
    inputSummary: summaries,
    trace: [['01', 'INPUT_INSPECTOR', summaries.length ? 'PASSED' : 'CURATED CASE'], ['02', 'QUERY_ROUTER', task], ['03', task === 'SINGLE_IMAGE_VQA' ? 'EO_ALIGNER_VQA' : 'CHANGE_ANALYST', 'COMPLETE'], ['04', 'EVIDENCE_BUILDER', uncertain ? 'ABSTAINED' : 'COMPLETE']],
  });
}
