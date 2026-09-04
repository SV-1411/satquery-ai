'use client';

import { FormEvent, useMemo, useRef, useState } from 'react';

/* Evidence masks are generated data URLs; next/image cannot optimize them. */
/* eslint-disable @next/next/no-img-element */

type Analysis = {
  runtime?: 'demo' | 'real';
  model_version?: string;
  evidence?: { source?: string; mask_png_base64?: string; visual_png_base64?: string; before_png_base64?: string; after_png_base64?: string; semantic_png_base64?: string; map_label?: string; analysis_path?: string; legend?: string[]; layers?: { id: string; title: string; status: string; meaning: string; png_base64?: string }[]; agreement?: number; changed_area_percent?: number; bbox_pixels?: number[] | null };
  ai_summary?: string;
  task: string;
  claim: string;
  where: string;
  magnitude: string;
  sensorCase: string;
  limit: string;
  confidence: number;
  decision: string;
  trace: string[][];
  inputSummary?: { name: string; format: string; modality?: string; modality_confidence?: number; modality_note?: string; bytes: number; width?: number; height?: number; bands?: number; status: string; note?: string }[];
};

type ModalityHint = 'AUTO' | 'OPTICAL' | 'SAR';

const cases = [
  {
    id: 'change-pair', code: 'MODE_001', title: 'CHANGE PAIR', type: 'TWO MATCHED OBSERVATIONS',
    sensors: ['OPTICAL / SAR', 'OPTICAL / SAR'], alignment: 'MATCHED PAIR REQUIRED', usable: 'TWO INPUTS', prompt: 'What changed between these two uploaded observations? Show the changed pixels.',
  },
  {
    id: 'single-observation', code: 'MODE_002', title: 'SINGLE OBSERVATION', type: 'ONE IMAGE / VISUAL SIGNALS',
    sensors: ['AUTO DETECT', 'NOT REQUIRED'], alignment: 'SINGLE INPUT', usable: 'ONE INPUT', prompt: 'Highlight water-covered regions and explain the evidence in simple words.',
  },
  {
    id: 'sensor-fusion', code: 'MODE_003', title: 'SENSOR FUSION', type: 'ONE OPTICAL + ONE SAR',
    sensors: ['OPTICAL', 'SAR'], alignment: 'MATCHED PAIR REQUIRED', usable: 'TWO INPUTS', prompt: 'Are the optical and SAR signals in agreement? Show the evidence from each sensor.',
  },
];

const initialAnalysis: Analysis = {
  task: 'READY_FOR_UPLOAD',
  claim: 'UPLOAD AN OBSERVATION TO START.',
  where: 'NO UPLOADED PIXELS ANALYSED YET.',
  magnitude: '2.4 KM² ESTIMATED / ±0.3 KM²',
  sensorCase: 'SAR: STRONG SUPPORT\nOPTICAL: PARTIAL AGREEMENT',
  limit: '31% OF OPTICAL PIXELS ARE CLOUD-OBSCURED.',
  confidence: 82,
  decision: 'TRIAGE-READY / HUMAN CONFIRMATION ADVISED',
  trace: [
    ['01', 'INPUT_INSPECTOR', 'PASSED'], ['02', 'QUERY_ROUTER', 'BI_TEMPORAL_CHANGE_VQA'],
    ['03', 'CHANGE_ANALYST', 'COMPLETE'], ['04', 'EVIDENCE_BUILDER', 'COMPLETE'],
  ],
};

export default function SatQueryConsole() {
  const [caseId, setCaseId] = useState(cases[0].id);
  const [mode, setMode] = useState<'fused' | 'optical' | 'sar'>('fused');
  const [query, setQuery] = useState('What changed, where, and which sensor supports it?');
  const [files, setFiles] = useState<File[]>([]);
  const [fileModalities, setFileModalities] = useState<ModalityHint[]>([]);
  const [analysis, setAnalysis] = useState<Analysis>({
    ...initialAnalysis,
    task: 'READY_FOR_UPLOAD',
    claim: 'UPLOAD AN OBSERVATION TO START.',
    where: 'NO UPLOADED PIXELS ANALYSED YET.',
    magnitude: 'NOT CALCULATED YET',
    sensorCase: 'AWAITING UPLOAD',
    limit: 'Upload one to four images, then ask a specific question.',
    confidence: 0,
    decision: 'READY TO ANALYSE',
    trace: [['01', 'UPLOAD_GATE', 'WAITING']],
  });
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [selectedLayerId, setSelectedLayerId] = useState('');
  const [uploadPreview, setUploadPreview] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const activeCase = useMemo(() => cases.find((item) => item.id === caseId) ?? cases[0], [caseId]);
  const evidenceMask = analysis.evidence?.mask_png_base64 ? `data:image/png;base64,${analysis.evidence.mask_png_base64}` : '';
  const mapPreview = analysis.evidence?.visual_png_base64 ? `data:image/png;base64,${analysis.evidence.visual_png_base64}` : '';
  const beforePreview = analysis.evidence?.before_png_base64 ? `data:image/png;base64,${analysis.evidence.before_png_base64}` : '';
  const afterPreview = analysis.evidence?.after_png_base64 ? `data:image/png;base64,${analysis.evidence.after_png_base64}` : '';
  const semanticOverlay = analysis.evidence?.semantic_png_base64 ? `data:image/png;base64,${analysis.evidence.semantic_png_base64}` : '';
  const evidenceLayers = analysis.evidence?.layers ?? [];
  const selectedLayer = evidenceLayers.find((layer) => layer.id === selectedLayerId && layer.png_base64);
  const displayedMap = selectedLayer?.png_base64 ? `data:image/png;base64,${selectedLayer.png_base64}` : mapPreview || uploadPreview;

  function resetForNewUpload(message: string) {
    setSelectedLayerId('');
    setUploadPreview('');
    setAnalysis({
      ...initialAnalysis,
      task: 'READY_FOR_UPLOAD',
      claim: 'NEW IMAGES READY FOR ANALYSIS.',
      where: 'NO EVIDENCE MAP UNTIL YOU RUN THIS UPLOAD.',
      magnitude: 'NOT CALCULATED YET',
      sensorCase: 'AWAITING UPLOAD',
      limit: 'The previous map was cleared. Run the new upload to generate fresh evidence.',
      confidence: 0,
      decision: 'READY TO ANALYSE',
      evidence: undefined,
      ai_summary: message,
      trace: [['01', 'NEW_UPLOAD', 'READY']],
    });
  }

  async function chooseFiles(list: FileList | null) {
    if (!list) return;
    const picked = Array.from(list).slice(0, 4);
    const invalid = picked.find((file) => !/\.(tif|tiff|png|jpe?g)$/i.test(file.name));
    if (invalid) {
      setError(`${invalid.name}: unsupported input. Use GeoTIFF, TIFF, PNG or JPEG.`);
      return;
    }
    setError('');
    setFiles(picked);
    setFileModalities(picked.map(() => 'AUTO'));
    resetForNewUpload('New images are ready. Click Run to replace the old map with evidence from these uploads.');
    const previewForm = new FormData();
    picked.forEach((file) => previewForm.append('files', file));
    picked.forEach((_, index) => previewForm.set(`modality${String.fromCharCode(65 + index)}`, 'AUTO'));
    try {
      const response = await fetch('/api/preview', { method: 'POST', body: previewForm });
      const payload = await response.json() as { previews?: string[] };
      if (response.ok && payload.previews?.[0]) setUploadPreview(`data:image/png;base64,${payload.previews[0]}`);
    } catch {
      // Preview is a convenience; the full analysis endpoint remains authoritative.
    }
  }

  async function runAnalysis(event: FormEvent) {
    event.preventDefault();
    setError('');
    setRunning(true);
    setSelectedLayerId('');
    setAnalysis((current) => ({ ...current, evidence: undefined, ai_summary: 'Analysing the newly uploaded images. The map will update when evidence is ready.', trace: [['01', 'UPLOADED_PIXELS', 'ANALYSING']] }));
    try {
      const form = new FormData();
      form.set('caseId', caseId);
      form.set('query', query);
      files.forEach((file) => form.append('files', file));
      if (files[0]) form.set('modalityA', fileModalities[0] ?? 'AUTO');
      if (files[1]) form.set('modalityB', fileModalities[1] ?? 'AUTO');
      if (files[2]) form.set('modalityC', fileModalities[2] ?? 'AUTO');
      if (files[3]) form.set('modalityD', fileModalities[3] ?? 'AUTO');
      const response = await fetch('/api/analyse', { method: 'POST', body: form });
      let result: Analysis & { error?: string };
      try {
        result = await response.json() as Analysis & { error?: string };
      } catch {
        if (response.status === 413) {
          throw new Error('This hosted page rejected the large raster upload before analysis. Open http://localhost:3000 for the running GPU service, or upload a smaller preview image here.');
        }
        throw new Error(`The upload gateway returned ${response.status} instead of an analysis result.`);
      }
      if (!response.ok) {
        if (result.error?.includes('Pair CRS values differ')) {
          throw new Error('These files use different map coordinate systems, so they are valid single-image samples but not a valid before/after pair. For change detection, use before_optical_A.tif with after_optical_corrected_D.tif from data/test/change_detection_demo.');
        }
        if (result.error?.includes('Pair dimensions differ')) {
          throw new Error('These files have different pixel grids. Use a matched pair with the same footprint and dimensions, or select a single-image/sensor-fusion mode.');
        }
        throw new Error(result.error ?? 'Analysis failed.');
      }
      setAnalysis(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Analysis failed.');
    } finally {
      setRunning(false);
    }
  }

  function downloadReport() {
    const report = [
      'SATQUERY AI / EVIDENCE REPORT', `SELECTED UI MODE: ${activeCase.title}`, `EXECUTED ANALYSIS: ${analysis.task}`,
      `INPUT COUNT: ${analysis.inputSummary?.length ?? files.length}`,
      `QUERY: ${query}`, '', `CLAIM: ${analysis.claim}`, `WHERE: ${analysis.where}`,
      `MAGNITUDE: ${analysis.magnitude}`, `SENSOR CASE: ${analysis.sensorCase}`,
      `EVIDENCE SCORE (NOT VALIDATED ACCURACY): ${analysis.confidence}/100`, `DECISION: ${analysis.decision}`,
      `LIMIT: ${analysis.limit}`, `AI SUMMARY: ${(analysis.ai_summary ?? 'Not available in demo mode.').replace(/^AI SUMMARY:\s*/i, '')}`, '', 'EXECUTION TRACE',
      ...analysis.trace.map((row) => row.join(' / ')), '', `GENERATED: ${new Date().toISOString()}`,
    ].join('\n');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([report], { type: 'text/plain' }));
    link.download = `satquery-${activeCase.id}-report.txt`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <main className="min-h-screen bg-white text-black">
      <header className="border-b-4 border-black px-4 py-3 md:px-6">
        <div className="flex items-start justify-between gap-6">
          <div><p className="kicker">SATQUERY AI / SENTRY</p><h1 className="display-title">EARTH, UNDER EVIDENCE.</h1></div>
          <div className="text-right font-mono text-[10px] leading-4 md:text-xs">
            <p>MODE: DISASTER TRIAGE</p><p>REGION: INDIA</p><p>STATUS: {running ? 'ANALYSING' : 'OPERATIONAL'}</p>
          </div>
        </div>
      </header>

      <section className="grid border-b-4 border-black lg:grid-cols-[280px_minmax(0,1fr)_360px]">
        <aside className="border-b-4 border-black p-4 lg:min-h-[720px] lg:border-b-0 lg:border-r-4">
          <p className="section-label">01 / ANALYSIS MODE</p>
          {cases.map((item) => (
            <button key={item.id} className={`case-card ${caseId === item.id ? 'case-card-active' : ''}`} onClick={() => { setCaseId(item.id); setQuery(item.prompt); setFiles([]); setFileModalities([]); resetForNewUpload('Choose images and run a new analysis for this mode.'); }}>
              <span className="font-mono text-[10px]">{item.code}</span><strong>{item.title}</strong><span>{item.type}</span>
            </button>
          ))}

          <button className="upload-box mt-5 w-full" onClick={() => fileInput.current?.click()} type="button">
            <span>+</span><strong>{files.length ? `${files.length} OBSERVATION${files.length > 1 ? 'S' : ''} LOADED` : 'ADD OBSERVATION'}</strong>
            <small>{files.length ? files.map((file) => file.name).join(' / ') : 'GEOTIFF / TIFF / PNG / JPEG'}</small>
          </button>
          <input ref={fileInput} className="sr-only" type="file" multiple accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(event) => chooseFiles(event.target.files)} />

          {files.length > 0 && <div className="modality-list">
            {files.map((file, index) => <label key={`${file.name}-${index}`} className="modality-row"><span>{file.name}</span><select value={fileModalities[index] ?? 'AUTO'} onChange={(event) => setFileModalities((current) => current.map((value, itemIndex) => itemIndex === index ? event.target.value as ModalityHint : value))}><option value="AUTO">AUTO DETECT</option><option value="OPTICAL">OPTICAL / MULTISPECTRAL</option><option value="SAR">SAR / RADAR</option></select></label>)}
          </div>}

          <dl className="metadata-grid mt-5">
            <div><dt>SENSOR A</dt><dd>{analysis.inputSummary?.[0]?.modality ?? fileModalities[0] ?? activeCase.sensors[0]}</dd></div>
            <div><dt>SENSOR B</dt><dd>{analysis.inputSummary?.[1]?.modality ?? fileModalities[1] ?? activeCase.sensors[1]}</dd></div>
            <div><dt>ALIGNMENT</dt><dd>{analysis.inputSummary?.length ? 'HEADER PASS' : activeCase.alignment}</dd></div>
            <div><dt>RASTER SIZE</dt><dd>{analysis.inputSummary?.[0]?.width ? `${analysis.inputSummary[0].width} × ${analysis.inputSummary[0].height}` : activeCase.usable}</dd></div>
          </dl>
          {error && <p className="error-box" role="alert">INPUT / {error}</p>}
        </aside>

        <section className="min-w-0 border-b-4 border-black lg:border-b-0 lg:border-r-4">
          <div className="flex items-center justify-between border-b-4 border-black px-4 py-3">
            <p className="section-label m-0">02 / EVIDENCE FIELD</p>
            <div className="flex gap-2 font-mono text-[10px]">
              {(['fused', 'optical', 'sar'] as const).map((item) => (
                <button key={item} className={`mode-button ${mode === item ? 'mode-button-active' : ''}`} onClick={() => setMode(item)}>{item.toUpperCase()}</button>
              ))}
            </div>
          </div>

          <div className={`evidence-map map-${mode} ${displayedMap ? 'has-uploaded-map' : ''} ${analysis.confidence < 50 ? 'map-abstained' : ''}`} aria-label={displayedMap ? 'Map generated from the newly uploaded imagery and selected evidence layer' : `${mode} evidence map showing probable change regions`}>
            {!displayedMap && <div className="empty-evidence">UPLOAD ONE TO FOUR IMAGES, ASK A QUESTION, THEN RUN THE ANALYSIS.<small>YOUR UPLOADED IMAGE AND ITS EVIDENCE LAYERS WILL APPEAR HERE.</small></div>}
            {displayedMap && <img key={displayedMap.slice(-40)} className="evidence-base" src={displayedMap} alt={selectedLayer ? `${selectedLayer.title} image layer` : 'Map rendered from the uploaded observation'} />}
            {!selectedLayer && semanticOverlay && <img className="semantic-overlay" src={semanticOverlay} alt="Colour-coded evidence derived from the uploaded pixels" />}
            {!selectedLayer && evidenceMask && <img className="evidence-mask" src={evidenceMask} alt="Evidence mask generated from uploaded pixels" />}
            <div className="north">N ↑</div><div className="scale">0 ━━━━━ 2 KM</div>
            <div className="map-caption">{selectedLayer ? `FILTER VIEW / ${selectedLayer.title}` : mapPreview ? analysis.evidence?.map_label ?? 'UPDATED FROM UPLOADED PIXELS' : uploadPreview ? 'UPLOADED PREVIEW / RUN TO GENERATE EVIDENCE' : 'UPLOAD AN IMAGE TO GENERATE EVIDENCE'}</div>
            {analysis.evidence?.legend?.length ? <div className="evidence-legend">{analysis.evidence.legend.map((item) => <span key={item}>{item}</span>)}</div> : null}
          </div>

          {(beforePreview || afterPreview) && <div className="comparison-strip" aria-label="Uploaded images used for this comparison">
            {beforePreview && <figure><img src={beforePreview} alt="First uploaded observation" /><figcaption>FIRST UPLOAD</figcaption></figure>}
            {afterPreview && <figure><img src={afterPreview} alt="Latest uploaded observation" /><figcaption>LATEST UPLOAD</figcaption></figure>}
          </div>}

          {analysis.evidence?.layers?.length ? <section className="evidence-layers" aria-label="Visual evidence layers derived from the upload">
            <div className="layer-heading"><strong>VISUAL EVIDENCE LAYERS</strong><span>FILTERS AVAILABLE FOR THIS UPLOAD</span></div>
            <p className="layer-instruction">SELECT ANY AVAILABLE LAYER TO INSPECT THAT EXACT FILTER IN THE MAIN EVIDENCE FIELD.</p>
            <div className="layer-grid">{analysis.evidence.layers.map((layer) => <button type="button" disabled={!layer.png_base64} onClick={() => setSelectedLayerId(layer.id)} className={`layer-card layer-${layer.status.toLowerCase().replaceAll('_', '-')} ${selectedLayerId === layer.id ? 'layer-card-selected' : ''}`} key={layer.id}>
              {layer.png_base64 ? <img src={`data:image/png;base64,${layer.png_base64}`} alt={`${layer.title}: ${layer.meaning}`} /> : <div className="layer-unavailable">NOT AVAILABLE<br />FOR THIS INPUT</div>}
              <div><strong>{layer.title}</strong><em>{layer.status.replaceAll('_', ' ')}</em><p>{layer.meaning}</p></div>
            </button>)}</div>
          </section> : null}

          {analysis.inputSummary?.length ? <section className="sensor-ledger" aria-label="Sensor inputs and analysis route">
            <div className="layer-heading"><strong>SENSOR INPUT LEDGER</strong><span>EVERY FILE RECEIVED BY THE ANALYSIS</span></div>
            <div className="sensor-grid">{analysis.inputSummary.map((input, index) => <article key={`${input.name}-${index}`}>
              <strong>INPUT {String(index + 1).padStart(2, '0')} / {input.modality ?? 'AUTO'}</strong>
              <p>{input.name}</p><span>{input.format} / {input.bands ?? '?'} BANDS / {input.width ?? '?'} × {input.height ?? '?'}</span>
              <em>{input.status} → {analysis.task}</em>
            </article>)}</div>
          </section> : null}

          <form className="border-t-4 border-black p-4" onSubmit={runAnalysis}>
            <label className="section-label" htmlFor="query">ASK THE OBSERVATION</label>
            <div className="mt-2 flex border-4 border-black">
              <input id="query" className="min-w-0 flex-1 px-3 py-4 font-mono text-xs uppercase outline-none" value={query} onChange={(event) => setQuery(event.target.value)} />
              <button className="run-button" type="submit" disabled={running}>{running ? 'RUNNING…' : 'RUN →'}</button>
            </div>
            <div className="suggestion-row">
              {['Highlight water-covered regions', 'What changed between these dates?', 'Are optical and SAR in agreement?'].map((prompt) => (
                <button type="button" key={prompt} onClick={() => setQuery(prompt)}>{prompt}</button>
              ))}
            </div>
          </form>
        </section>

        <aside className="p-4" aria-live="polite">
          <div className="flex items-center justify-between gap-4"><p className="section-label">03 / EVIDENCE CONTRACT</p><span className="task-stamp">{analysis.runtime ? `${analysis.runtime.toUpperCase()} / ${analysis.task}` : 'READY / NO UPLOAD'}</span></div>
          <div className="claim-box"><p>CLAIM</p><h2>{analysis.claim}</h2></div>
          <div className="ai-summary"><p>AI SUMMARY</p><span>{analysis.ai_summary ?? 'Run an upload to generate a grounded summary from the observation evidence.'}</span></div>
          {analysis.evidence?.analysis_path && <div className="analysis-path"><p>HOW THIS WAS DERIVED</p><span>{analysis.evidence.analysis_path}</span></div>}
          <div className="contract-row"><span>WHERE</span><p>{analysis.where}</p></div>
          <div className="contract-row"><span>HOW MUCH</span><p>{analysis.magnitude}</p></div>
          <div className="contract-row"><span>SENSOR CASE</span><p className="whitespace-pre-line">{analysis.sensorCase}</p></div>
          <div className="contract-row"><span>LIMIT / NEXT OBSERVATION</span><p>{analysis.limit}</p></div>
          <div className="confidence-block">
            <div><span>EVIDENCE SCORE</span><strong>{analysis.confidence}</strong></div>
            <div className="confidence-meter"><i style={{ width: `${analysis.confidence}%` }} /></div><p>INTERNAL SIGNAL AGREEMENT — NOT VALIDATED ACCURACY</p><p>{analysis.decision}</p>
          </div>
          <details className="trace-box" open><summary>OBSERVABLE EXECUTION TRACE</summary>
            {analysis.trace.map(([step, tool, status]) => <div className="trace-row" key={`${step}-${tool}`}><span>{step}</span><strong>{tool}</strong><em>{status}</em></div>)}
          </details>
          <p className="mt-3 font-mono text-[10px] uppercase">MODEL / {analysis.model_version ?? 'LOCAL GPU MODEL WHEN ANALYSIS RUNS'}</p>
          <button className="report-button" type="button" onClick={downloadReport}>DOWNLOAD EVIDENCE REPORT ↓</button>
        </aside>
      </section>
      <footer className="flex flex-col justify-between gap-2 px-4 py-3 font-mono text-[10px] uppercase md:flex-row md:px-6">
        <span>Evidence is a claim with a location, a source and a limit.</span><span>SatQuery Sentry / Prototype 0.1</span>
      </footer>
    </main>
  );
}
