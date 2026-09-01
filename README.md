# SatQuery AI — SENTRY

SatQuery AI is an evidence-first, agentic vision-language assistant for remote-sensing imagery. It accepts a single optical/SAR image, a co-registered optical–SAR pair, or a bi-temporal pair and routes a natural-language question to the appropriate specialist workflow.

## Start here

The complete project explanation and operator guide is here:

[`docs/SatQuery_AI_Operator_and_Implementation_Handbook.docx`](docs/SatQuery_AI_Operator_and_Implementation_Handbook.docx)

It covers:

- What the problem statement asks for
- What SatQuery AI does and what is innovative about it
- The architecture and agentic orchestration flow
- Every UI control and expected result
- Local run and API testing instructions
- Dataset and benchmark mapping
- Current prototype boundaries
- Deployment and GitHub workflows
- Production implementation roadmap
- Evaluation criteria and troubleshooting

## Run locally

```powershell
Set-Location D:\satquery-ai
npm install
npm run dev
```

Open <http://localhost:3000>.

Validation commands:

```powershell
npm run lint
npm run build
```

## Current demo

The public shareable demo is deployed at <https://satquery-ai-public.vercel.app>.

The owner-only Sites deployment is also available at <https://satquery-sentry.shrutikaverma22.chatgpt.site> for internal review.

The UI uses a black-and-white brutalist control system and a full-colour satellite-style evidence basemap. The prototype currently performs real multipart upload validation and raster-header inspection. Its answer claims are deterministic representative outputs intended to demonstrate the evidence contract and orchestration surface while specialist remote-sensing models are integrated.

## Main implementation files

- `app/satquery-console.tsx` — interactive console, cases, modes, upload handling and report download.
- `app/api/analyse/route.ts` — multipart validation, TIFF/PNG/JPEG inspection and task routing.
- `app/globals.css` — interface styling and evidence map layers.
- `public/evidence-map.png` — satellite-style evidence basemap.
- `docs/SatQuery_AI_Operator_and_Implementation_Handbook.docx` — full documentation.
