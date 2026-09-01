import { NextResponse } from 'next/server';

type AnalyseRequest = {
  caseId?: string;
  query?: string;
  inputNames?: string[];
};

const supportedExtensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];

function classify(query: string, fileCount: number) {
  const q = query.toLowerCase();
  if (/(changed|change|between|increased|decreased|before|after)/.test(q) && fileCount >= 2) {
    return 'BI_TEMPORAL_CHANGE_VQA';
  }
  if (/(highlight|locate|where|mark|region)/.test(q)) return 'TEXT_GUIDED_GROUNDING';
  if (/(sar|radar|optical|sensor|together|fuse)/.test(q) && fileCount >= 2) {
    return 'OPTICAL_SAR_FUSION';
  }
  return 'SINGLE_IMAGE_VQA';
}

export async function POST(request: Request) {
  let body: AnalyseRequest;
  try {
    body = (await request.json()) as AnalyseRequest;
  } catch {
    return NextResponse.json({ error: 'The request body is not valid JSON.' }, { status: 400 });
  }

  const query = body.query?.trim() ?? '';
  const inputNames = body.inputNames ?? [];
  const fileCount = Math.max(inputNames.length, body.caseId ? 2 : 0);

  if (query.length < 5) {
    return NextResponse.json({ error: 'Write a specific observation question.' }, { status: 422 });
  }

  const invalidFile = inputNames.find((name) => {
    const lower = name.toLowerCase();
    return !supportedExtensions.some((extension) => lower.endsWith(extension));
  });
  if (invalidFile) {
    return NextResponse.json(
      { error: `${invalidFile} is not a supported GeoTIFF, TIFF, PNG or JPEG input.` },
      { status: 415 },
    );
  }

  const task = classify(query, fileCount);
  const uncertain = /(cloud|uncertain|disagree|insufficient)/i.test(query);

  await new Promise((resolve) => setTimeout(resolve, 650));

  return NextResponse.json({
    task,
    claim: uncertain
      ? 'THE AVAILABLE OBSERVATIONS DO NOT SUPPORT A RELIABLE CONCLUSION.'
      : task === 'TEXT_GUIDED_GROUNDING'
        ? 'TWO WATER-COVERED REGIONS MATCH THE REQUEST.'
        : task === 'SINGLE_IMAGE_VQA'
          ? 'WATER, BUILT-UP LAND AND ROAD CORRIDORS ARE VISIBLE.'
          : 'NEWLY INUNDATED LAND IS LIKELY PRESENT.',
    where: uncertain ? 'NO REGION RELEASED' : 'NORTH-EAST LOW-LYING REGION / ZONES A + B',
    magnitude: uncertain ? 'NOT ESTIMATED' : '2.4 KM² ESTIMATED / ±0.3 KM²',
    sensorCase: uncertain
      ? 'OPTICAL AND SAR EVIDENCE DISAGREE'
      : 'SAR: STRONG SUPPORT\nOPTICAL: PARTIAL AGREEMENT',
    limit: uncertain
      ? 'ACQUIRE A LATER CLOUD-FREE OPTICAL IMAGE OR REQUEST ANALYST REVIEW.'
      : '31% OF OPTICAL PIXELS ARE CLOUD-OBSCURED.',
    confidence: uncertain ? 43 : task === 'SINGLE_IMAGE_VQA' ? 76 : 82,
    decision: uncertain ? 'INSUFFICIENT EVIDENCE' : 'TRIAGE-READY / HUMAN CONFIRMATION ADVISED',
    trace: [
      ['01', 'INPUT_INSPECTOR', invalidFile ? 'REJECTED' : 'PASSED'],
      ['02', 'QUERY_ROUTER', task],
      ['03', task === 'SINGLE_IMAGE_VQA' ? 'EO_ALIGNER_VQA' : 'CHANGE_ANALYST', 'COMPLETE'],
      ['04', 'EVIDENCE_BUILDER', uncertain ? 'ABSTAINED' : 'COMPLETE'],
    ],
  });
}
