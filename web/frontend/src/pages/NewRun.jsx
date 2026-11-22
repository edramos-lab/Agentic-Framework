import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { startPipeline, uploadFiles } from "../api";
import {
  Alert,
  Button,
  Checkbox,
  FileInput,
  Input,
  PageHeader,
  Select,
  Textarea,
  IconPlay,
} from "../components/ui";

const GPU_OPTIONS = ["T4", "L4", "G4", "H100", "A100"];
const PROVIDERS = ["openai", "anthropic", "openrouter"];

function Section({ title, description, children }) {
  return (
    <fieldset className="card p-5 sm:p-6">
      <legend className="float-left mb-4 w-full text-base font-semibold text-slate-900">
        {title}
        {description && (
          <span className="mt-0.5 block text-sm font-normal text-slate-600">
            {description}
          </span>
        )}
      </legend>
      <div className="clear-both">{children}</div>
    </fieldset>
  );
}

export default function NewRun() {
  const navigate = useNavigate();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  // --- Form state ---
  const [idea, setIdea] = useState({
    title: "",
    abstract_draft: "",
    research_questions: "",
    contributions: "",
    methodology_sketch: "",
    keywords: "",
  });

  const [guideline, setGuideline] = useState({
    name: "IEEE",
    page_limit: 8,
    format_notes: "",
    citation_style: "ieee",
    abstract_word_limit: 250,
    sections_required: "Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion",
  });

  // The dataset the experiments use; the defaults are the leukemia study (blood smears from Kaggle)
  const [dataset, setDataset] = useState({
    domain: "microscopy",
    source: "kaggle",
    id: "mehradaria/leukemia",
    split: "official",
    group_regex: "",
  });

  const [colab, setColab] = useState({
    enabled: true,
    gpu_type: "T4",
    session_name: "paper-orchestra",
    keep_session: false,
    pip_packages: "",
  });

  const [jetson, setJetson] = useState({
    enabled: false,
    host: "",
    user: "jetson",
    ssh_key: "~/.ssh/id_ed25519",
    remote_dir: "/home/jetson/paper_orchestra",
  });

  const [experiment, setExperiment] = useState({
    datasets: "",
    models: "",
    metrics: "",
    baselines: "",
    timeout_minutes: 120,
  });

  const [llm, setLlm] = useState({
    provider: "openai",
    model: "gpt-4o",
    temperature: 0.3,
    max_tokens: 16384,
  });

  const [templateFile, setTemplateFile] = useState(null);
  const [scriptFiles, setScriptFiles] = useState([]);
  const [styleFile, setStyleFile] = useState(null);

  // --- handlers ---

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);

    try {
      // Upload template if provided
      let templateFilename = "template.tex";
      if (templateFile) {
        await uploadFiles([templateFile]);
        templateFilename = templateFile.name;
      }
      // Upload script files
      if (scriptFiles.length > 0) {
        await uploadFiles(scriptFiles);
      }
      // Upload the author's own writing, so the writing agent can imitate it
      let styleFilename = "";
      if (styleFile) {
        await uploadFiles([styleFile]);
        styleFilename = styleFile.name;
      }

      const payload = {
        idea: {
          ...idea,
          research_questions: idea.research_questions
            .split("\n")
            .map((s) => s.trim())
            .filter(Boolean),
          contributions: idea.contributions
            .split("\n")
            .map((s) => s.trim())
            .filter(Boolean),
          keywords: idea.keywords
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
        },
        guideline: {
          ...guideline,
          sections_required: guideline.sections_required
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
        },
        experiment_requirements: {
          datasets: experiment.datasets
            .split("\n")
            .map((s) => s.trim())
            .filter(Boolean),
          models: experiment.models
            .split("\n")
            .map((s) => s.trim())
            .filter(Boolean),
          metrics: experiment.metrics
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          baselines: experiment.baselines
            .split("\n")
            .map((s) => s.trim())
            .filter(Boolean),
          timeout_minutes: experiment.timeout_minutes,
          dataset: {
            domain: dataset.domain,
            source: dataset.source,
            id: dataset.id.trim(),
            split: dataset.split,
            group_regex: dataset.group_regex.trim(),
          },
          // Training goes to Colab; edge benchmarks go to the Jetson when one is set up
          backend: "colab",
          jetson: {
            host: jetson.enabled ? jetson.host.trim() : "",
            user: jetson.user,
            ssh_key: jetson.ssh_key,
            remote_dir: jetson.remote_dir,
          },
          colab: {
            enabled: colab.enabled,
            gpu_type: colab.gpu_type,
            session_name: colab.session_name,
            keep_session: colab.keep_session,
            pip_packages: colab.pip_packages
              .split("\n")
              .map((s) => s.trim())
              .filter(Boolean),
          },
        },
        llm,
        latex_template_filename: templateFilename,
        style_samples_filename: styleFilename,
      };

      const { run_id } = await startPipeline(payload);
      navigate(`/run/${run_id}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const up = (setter) => (field) => (e) =>
    setter((prev) => ({ ...prev, [field]: e.target.value }));
  const upIdea = up(setIdea);
  const upGuide = up(setGuideline);
  const upDataset = up(setDataset);
  const upColab = up(setColab);
  const upJetson = up(setJetson);
  const upExp = up(setExperiment);
  const upLlm = up(setLlm);

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <PageHeader
        title="New Pipeline Run"
        description="Describe your research idea and constraints. The agents will draft, run experiments, and compile the paper."
      />

      {error && <Alert>{error}</Alert>}

      {/* Research Idea */}
      <Section title="Research Idea">
        <div className="space-y-4">
          <Input
            label="Title"
            required
            value={idea.title}
            onChange={upIdea("title")}
            placeholder="e.g. Vision Transformer Pruning for Edge Deployment"
          />
          <Textarea
            label="Abstract Draft"
            value={idea.abstract_draft}
            onChange={upIdea("abstract_draft")}
            placeholder="A rough abstract or high-level summary…"
          />
          <Textarea
            label="Research Questions (one per line)"
            value={idea.research_questions}
            onChange={upIdea("research_questions")}
            placeholder={"RQ1: Can structured pruning reduce ViT inference by 50%?\nRQ2: ..."}
          />
          <Textarea
            label="Key Contributions (one per line)"
            value={idea.contributions}
            onChange={upIdea("contributions")}
          />
          <Textarea
            label="Methodology Sketch"
            value={idea.methodology_sketch}
            onChange={upIdea("methodology_sketch")}
          />
          <Input
            label="Keywords (comma-separated)"
            value={idea.keywords}
            onChange={upIdea("keywords")}
            placeholder="vision transformers, pruning, edge AI"
          />
        </div>
      </Section>

      {/* Conference Guideline */}
      <Section title="Conference Guideline">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Conference Name"
            value={guideline.name}
            onChange={upGuide("name")}
          />
          <Input
            label="Page Limit"
            type="number"
            value={guideline.page_limit}
            onChange={(e) =>
              setGuideline((p) => ({ ...p, page_limit: +e.target.value }))
            }
          />
          <Input
            label="Citation Style"
            value={guideline.citation_style}
            onChange={upGuide("citation_style")}
          />
          <Input
            label="Abstract Word Limit"
            type="number"
            value={guideline.abstract_word_limit}
            onChange={(e) =>
              setGuideline((p) => ({
                ...p,
                abstract_word_limit: +e.target.value,
              }))
            }
          />
        </div>
        <div className="mt-4">
          <Input
            label="Required Sections (comma-separated)"
            value={guideline.sections_required}
            onChange={upGuide("sections_required")}
          />
        </div>
        <div className="mt-4">
          <Textarea
            label="Format Notes"
            value={guideline.format_notes}
            onChange={upGuide("format_notes")}
            placeholder="Two-column, 10pt font…"
          />
        </div>
      </Section>

      {/* LaTeX Template Upload */}
      <Section title="LaTeX Template">
        <FileInput
          label="Template file"
          hint="Optional. Accepts .tex, .sty, .cls, .bst. Defaults to the IEEE template."
          accept=".tex,.sty,.cls,.bst"
          onChange={(e) => setTemplateFile(e.target.files[0] || null)}
        />
      </Section>

      {/* Experiment Configuration */}
      <Section title="Experiment Configuration">
        <div className="space-y-4">
          <Textarea
            label="Datasets (one per line)"
            value={experiment.datasets}
            onChange={upExp("datasets")}
            placeholder={"ImageNet-1K\nCIFAR-100"}
          />
          <Textarea
            label="Models (one per line)"
            value={experiment.models}
            onChange={upExp("models")}
            placeholder={"ViT-B/16\nDeiT-Small"}
          />
          <Input
            label="Metrics (comma-separated)"
            value={experiment.metrics}
            onChange={upExp("metrics")}
            placeholder="top1_accuracy, top5_accuracy, latency_ms, flops"
          />
          <Textarea
            label="Baselines (one per line)"
            value={experiment.baselines}
            onChange={upExp("baselines")}
          />
          <Input
            label="Timeout (minutes)"
            type="number"
            value={experiment.timeout_minutes}
            onChange={(e) =>
              setExperiment((p) => ({
                ...p,
                timeout_minutes: +e.target.value,
              }))
            }
          />
          <FileInput
            label="Experiment scripts"
            hint="Optional. Accepts .py, .sh, .r, and training_history.json (per-epoch train_loss, val_loss, train_acc, val_acc) for training curves"
            multiple
            accept=".py,.sh,.r,.json"
            onChange={(e) => setScriptFiles([...e.target.files])}
          />
        </div>
      </Section>

      {/* Dataset */}
      <Section
        title="Dataset"
        description="Which data the experiments use and which recipe runs on them. Kaggle and Zenodo tokens are kept on the API Keys page."
      >
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <Select label="Kind of images" value={dataset.domain} onChange={upDataset("domain")}>
            <option value="microscopy">Blood-smear microscopy (cells, stains)</option>
            <option value="photos">Photographs (person, background)</option>
          </Select>
          <Select label="Source" value={dataset.source} onChange={upDataset("source")}>
            <option value="kaggle">Kaggle</option>
            <option value="zenodo">Zenodo</option>
          </Select>
          <Input
            label={dataset.source === "zenodo" ? "Zenodo record id" : "Kaggle dataset (owner/name)"}
            value={dataset.id}
            onChange={upDataset("id")}
          />
          <Select label="Train/test split" value={dataset.split} onChange={upDataset("split")}>
            <option value="official">The dataset's own train/ and test/ folders</option>
            <option value="random">Random, stratified</option>
          </Select>
          <Input
            label="Capture session in the file name (regex, optional)"
            value={dataset.group_regex}
            onChange={upDataset("group_regex")}
          />
        </div>
      </Section>

      {/* Colab Settings */}
      <Section title="Google Colab (Experiment Runner)">
        <div className="space-y-4">
          <Checkbox
            label="Run experiments on Google Colab"
            checked={colab.enabled}
            onChange={(e) =>
              setColab((p) => ({ ...p, enabled: e.target.checked }))
            }
          />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <Select
              label="GPU"
              value={colab.gpu_type}
              onChange={(e) =>
                setColab((p) => ({ ...p, gpu_type: e.target.value }))
              }
            >
              {GPU_OPTIONS.map((g) => (
                <option key={g}>{g}</option>
              ))}
            </Select>
            <Input
              label="Session Name"
              value={colab.session_name}
              onChange={upColab("session_name")}
            />
            <div className="sm:pt-6">
              <Checkbox
                label="Keep session alive"
                checked={colab.keep_session}
                onChange={(e) =>
                  setColab((p) => ({ ...p, keep_session: e.target.checked }))
                }
              />
            </div>
          </div>
          <Textarea
            label="Pip Packages (one per line)"
            value={colab.pip_packages}
            onChange={upColab("pip_packages")}
            placeholder={"torch>=2.1\ntimm\nwandb"}
          />
        </div>
      </Section>

      {/* Jetson edge deployment */}
      <Section
        title="Jetson Orin Nano (Edge Deployment)"
        description="Colab trains the model; the Jetson benchmarks it (PyTorch, ONNX, TensorRT FP32/FP16/INT8) and the results go into the paper."
      >
        <div className="space-y-4">
          <Checkbox
            label="Benchmark on a Jetson"
            checked={jetson.enabled}
            onChange={(e) => setJetson((p) => ({ ...p, enabled: e.target.checked }))}
          />
          {jetson.enabled && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Input
                label="IP address or hostname"
                required
                value={jetson.host}
                onChange={upJetson("host")}
                placeholder="100.96.228.9"
              />
              <Input label="SSH user" value={jetson.user} onChange={upJetson("user")} />
              <Input
                label="SSH key path"
                value={jetson.ssh_key}
                onChange={upJetson("ssh_key")}
                hint="Path inside the backend. With Docker, set JETSON_SSH_KEY before starting compose."
              />
              <Input
                label="Remote directory"
                value={jetson.remote_dir}
                onChange={upJetson("remote_dir")}
              />
            </div>
          )}
        </div>
      </Section>

      {/* Writing voice */}
      <Section
        title="Your writing voice"
        description="The writing agent imitates your own prose, following the thesis-voice-coach rules."
      >
        <FileInput
          label="Writing samples"
          hint="Optional. A .txt or .tex file with a few paragraphs you wrote yourself."
          accept=".txt,.tex,.md"
          onChange={(e) => setStyleFile(e.target.files[0] || null)}
        />
      </Section>

      {/* LLM Settings */}
      <Section title="LLM Configuration">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Select
            label="Provider"
            value={llm.provider}
            onChange={(e) =>
              setLlm((p) => ({ ...p, provider: e.target.value }))
            }
          >
            {PROVIDERS.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </Select>
          <Input
            label="Model"
            value={llm.model}
            onChange={upLlm("model")}
            placeholder="gpt-4o / claude-sonnet-5-5 / (openrouter) anthropic/claude-sonnet-5.5"
          />
          <Input
            label="Temperature"
            type="number"
            step="0.1"
            min="0"
            max="2"
            value={llm.temperature}
            onChange={(e) =>
              setLlm((p) => ({ ...p, temperature: +e.target.value }))
            }
          />
          <Input
            label="Max Tokens"
            type="number"
            value={llm.max_tokens}
            onChange={(e) =>
              setLlm((p) => ({ ...p, max_tokens: +e.target.value }))
            }
          />
        </div>
      </Section>

      {/* Submit */}
      <div className="flex justify-end">
        <Button type="submit" loading={submitting} className="px-6">
          {!submitting && <IconPlay />}
          {submitting ? "Starting…" : "Start pipeline"}
        </Button>
      </div>
    </form>
  );
}
