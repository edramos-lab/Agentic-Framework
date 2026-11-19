import React, { useEffect, useState } from "react";
import { listKeys, setKey, deleteKey } from "../api";
import {
  Alert,
  Button,
  Input,
  PageHeader,
  Select,
  IconCheck,
  IconTrash,
} from "../components/ui";

const KEY_NAMES = [
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "SEMANTIC_SCHOLAR_API_KEY",
  "WANDB_API_KEY",
  "HF_TOKEN",
];

export default function KeysManager() {
  const [keys, setKeys] = useState([]);
  const [name, setName] = useState("");
  const [value, setValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [pendingDelete, setPendingDelete] = useState(null);

  const refresh = () =>
    listKeys()
      .then(setKeys)
      .catch((err) => setError(err.message));
  useEffect(() => {
    refresh();
  }, []);

  const handleAdd = async (e) => {
    e.preventDefault();
    if (!name || !value) return;
    setSaving(true);
    setError("");
    try {
      await setKey(name, value);
      setName("");
      setValue("");
      refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const confirmDelete = async (n) => {
    setPendingDelete(null);
    try {
      await deleteKey(n);
      refresh();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="max-w-2xl space-y-6">
      <PageHeader
        title="API Key Vault"
        description="Keys are encrypted at rest with Fernet and injected into the pipeline process as environment variables. They are never written to config files."
      />

      {error && <Alert>{error}</Alert>}

      <form onSubmit={handleAdd} className="card space-y-4 p-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <Select
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          >
            <option value="">Select a key…</option>
            {KEY_NAMES.map((k) => (
              <option key={k}>{k}</option>
            ))}
          </Select>
          <Input
            label="Value"
            type="password"
            autoComplete="off"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="sk-…"
          />
        </div>
        <div className="flex justify-end">
          <Button type="submit" loading={saving} disabled={!name || !value}>
            {saving ? "Saving…" : "Save key"}
          </Button>
        </div>
      </form>

      <section aria-labelledby="stored-keys">
        <h2 id="stored-keys" className="mb-2 text-sm font-semibold text-slate-700">
          Stored keys
        </h2>
        {keys.length === 0 ? (
          <p className="card px-5 py-6 text-center text-sm text-slate-600">
            No keys stored yet. Add one above to run pipelines.
          </p>
        ) : (
          <ul className="card divide-y divide-slate-200">
            {keys.map((k) => (
              <li
                key={k.name}
                className="flex min-h-[56px] items-center justify-between gap-3 px-5 py-2 text-sm"
              >
                <div className="flex min-w-0 items-center gap-3">
                  <span className="truncate font-mono text-slate-900">
                    {k.name}
                  </span>
                  <span className="inline-flex items-center gap-1 text-xs font-medium text-green-700">
                    <IconCheck className="h-3.5 w-3.5" /> Stored
                  </span>
                </div>
                {pendingDelete === k.name ? (
                  <div className="flex items-center gap-1">
                    <Button
                      variant="danger"
                      onClick={() => confirmDelete(k.name)}
                      aria-label={`Confirm removing ${k.name}`}
                    >
                      Confirm remove
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => setPendingDelete(null)}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="danger"
                    onClick={() => setPendingDelete(k.name)}
                    aria-label={`Remove ${k.name}`}
                  >
                    <IconTrash />
                    Remove
                  </Button>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
