import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listRuns } from "../api";
import {
  Alert,
  EmptyState,
  PageHeader,
  Skeleton,
  StatusBadge,
  IconArrowRight,
} from "../components/ui";

export default function History() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Run History"
        description="Every pipeline run, newest activity first."
      />

      {error && <Alert>Couldn't load runs: {error}</Alert>}

      {loading ? (
        <div className="card space-y-4 p-5" aria-busy="true" aria-label="Loading runs">
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-5 w-1/2" />
          <Skeleton className="h-5 w-3/4" />
        </div>
      ) : runs.length === 0 && !error ? (
        <EmptyState title="No runs yet">
          <Link to="/" className="font-medium text-brand-700 hover:underline">
            Start your first pipeline run
          </Link>
        </EmptyState>
      ) : (
        <ul className="card divide-y divide-slate-200 overflow-hidden">
          {runs.map((r) => (
            <li key={r.run_id}>
              <Link
                to={`/run/${r.run_id}`}
                className="flex min-h-[56px] items-center justify-between gap-3 px-5 py-3 transition-colors duration-150 hover:bg-slate-50"
              >
                <div className="min-w-0">
                  <p className="truncate font-mono text-sm text-slate-900">
                    {r.run_id}
                  </p>
                  {r.started_at && (
                    <p className="text-xs text-slate-500">
                      {new Date(r.started_at).toLocaleString()}
                    </p>
                  )}
                </div>
                <div className="flex shrink-0 items-center gap-3">
                  <StatusBadge status={r.status} />
                  <IconArrowRight className="h-4 w-4 text-slate-400" />
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
