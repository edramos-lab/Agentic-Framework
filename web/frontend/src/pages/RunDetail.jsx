import React, { useEffect, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRun, listArtifacts, connectPipelineWS } from "../api";
import PipelineBoard from "../components/PipelineBoard";
import {
  Alert,
  PageHeader,
  Skeleton,
  StatusBadge,
  IconArrowLeft,
  IconDownload,
} from "../components/ui";

const ACTIVE = new Set(["pending", "queued", "running"]);

export default function RunDetail() {
  const { id } = useParams();
  const [run, setRun] = useState(null);
  const [artifacts, setArtifacts] = useState([]);
  const [logs, setLogs] = useState([]);
  const [loadError, setLoadError] = useState("");
  const logRef = useRef(null);

  const refresh = () =>
    getRun(id)
      .then((r) => {
        setRun(r);
        setLoadError("");
        return r;
      })
      .catch((err) => setLoadError(err.message));

  // The snapshot is the source of truth: polling keeps every tab and reload correct.
  useEffect(() => {
    let timer;
    let stopped = false;
    const tick = async () => {
      const r = await refresh();
      if (stopped) return;
      if (!r || ACTIVE.has(r.status)) {
        timer = setTimeout(tick, 1500);
      } else {
        listArtifacts(id).then(setArtifacts).catch(console.error);
      }
    };
    tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // The WebSocket only carries log lines; any agent event also nudges a refresh.
  useEffect(() => {
    if (!id) return undefined;
    const ws = connectPipelineWS(id, (event) => {
      if (event.type === "log") {
        setLogs((prev) => [...prev, event.message]);
      } else if (event.type === "agent_update" || event.type === "agent_step") {
        refresh();
      }
    });
    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // Keep the newest log line in view
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  if (loadError) {
    return (
      <div className="space-y-4">
        <Alert>Couldn't load run {id}: {loadError}</Alert>
        <Link to="/history" className="inline-flex items-center gap-1 text-sm font-medium text-brand-700 hover:underline">
          <IconArrowLeft /> All runs
        </Link>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="space-y-4" aria-busy="true" aria-label="Loading run">
        <Skeleton className="h-8 w-1/3" />
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Skeleton className="h-16" />
          <Skeleton className="h-16" />
          <Skeleton className="h-16" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader title={run.idea_title || `Run ${id}`} description={run.idea_title ? `Run ${id}` : undefined}>
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={run.status} />
          {run.status === "completed" && (
            <a
              href={`/api/pipeline/runs/${id}/overleaf.zip`}
              download
              className="inline-flex min-h-[44px] items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white transition-colors duration-150 hover:bg-brand-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
            >
              <IconDownload />
              Download for Overleaf (.zip)
            </a>
          )}
        </div>
      </PageHeader>

      {run.error && <Alert>{run.error}</Alert>}

      <PipelineBoard agents={run.agents || []} runStatus={run.status} />

      {logs.length > 0 && (
        <section aria-labelledby="log-h">
          <h2 id="log-h" className="mb-3 text-lg font-semibold">
            Log
          </h2>
          <div
            ref={logRef}
            role="log"
            tabIndex={0}
            aria-label="Pipeline log"
            className="max-h-72 space-y-1 overflow-y-auto rounded-xl bg-slate-900 p-4 font-mono text-xs leading-relaxed text-slate-100"
          >
            {logs.map((l, i) => (
              <div key={i} className="break-words">
                {l}
              </div>
            ))}
          </div>
        </section>
      )}

      {artifacts.length > 0 && (
        <section aria-labelledby="files-h">
          <h2 id="files-h" className="mb-3 text-lg font-semibold">
            Output files
          </h2>
          <ul className="card divide-y divide-slate-200">
            {artifacts.map((a) => (
              <li
                key={a.filename}
                className="flex min-h-[52px] items-center justify-between gap-3 px-5 py-2 text-sm"
              >
                <span className="min-w-0 truncate">{a.filename}</span>
                <div className="flex shrink-0 items-center gap-4">
                  <span className="text-xs text-slate-500 tabular-nums">
                    {(a.size / 1024).toFixed(1)} KB
                  </span>
                  <a
                    href={a.url}
                    download
                    className="inline-flex min-h-[44px] items-center gap-1.5 font-medium text-brand-700 hover:underline"
                  >
                    <IconDownload />
                    Download
                  </a>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <Link
        to="/history"
        className="inline-flex min-h-[44px] items-center gap-1.5 text-sm font-medium text-brand-700 hover:underline"
      >
        <IconArrowLeft /> All runs
      </Link>
    </div>
  );
}
