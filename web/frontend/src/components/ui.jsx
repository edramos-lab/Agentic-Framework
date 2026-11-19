import React, { useId } from "react";

/* ---------- Icons (Lucide-style, 24px grid, 2px stroke) ---------- */

function Icon({ children, className = "h-4 w-4", ...props }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
      {...props}
    >
      {children}
    </svg>
  );
}

export const IconLogo = (p) => (
  <Icon {...p}>
    <path d="M9 18V5l12-2v13" />
    <circle cx="6" cy="18" r="3" />
    <circle cx="18" cy="16" r="3" />
  </Icon>
);
export const IconPlay = (p) => (
  <Icon {...p}>
    <polygon points="6 3 20 12 6 21 6 3" />
  </Icon>
);
export const IconClock = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </Icon>
);
export const IconSpinner = ({ className = "h-4 w-4", ...p }) => (
  <Icon className={`${className} animate-spin`} {...p}>
    <path d="M21 12a9 9 0 1 1-6.2-8.55" />
  </Icon>
);
export const IconCheck = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="10" />
    <path d="m9 12 2 2 4-4" />
  </Icon>
);
export const IconX = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="10" />
    <path d="m15 9-6 6M9 9l6 6" />
  </Icon>
);
export const IconAlert = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 8v4M12 16h.01" />
  </Icon>
);
export const IconDownload = (p) => (
  <Icon {...p}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
  </Icon>
);
export const IconArrowLeft = (p) => (
  <Icon {...p}>
    <path d="m12 19-7-7 7-7M19 12H5" />
  </Icon>
);
export const IconArrowRight = (p) => (
  <Icon {...p}>
    <path d="M5 12h14M12 5l7 7-7 7" />
  </Icon>
);
export const IconTrash = (p) => (
  <Icon {...p}>
    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </Icon>
);
export const IconKey = (p) => (
  <Icon {...p}>
    <circle cx="7.5" cy="15.5" r="5.5" />
    <path d="m21 2-9.6 9.6M15.5 7.5l3 3L22 7l-3-3" />
  </Icon>
);

export const IconList = (p) => (
  <Icon {...p}>
    <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />
  </Icon>
);
export const IconFlask = (p) => (
  <Icon {...p}>
    <path d="M9 3h6M10 3v6.5L4.5 19a1.5 1.5 0 0 0 1.3 2.2h12.4a1.5 1.5 0 0 0 1.3-2.2L14 9.5V3M7.5 14h9" />
  </Icon>
);
export const IconBook = (p) => (
  <Icon {...p}>
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5v14ZM4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5" />
  </Icon>
);
export const IconChart = (p) => (
  <Icon {...p}>
    <path d="M3 3v18h18M7 15l4-4 3 3 5-6" />
  </Icon>
);
export const IconPen = (p) => (
  <Icon {...p}>
    <path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
  </Icon>
);
export const IconRefresh = (p) => (
  <Icon {...p}>
    <path d="M21 12a9 9 0 0 0-15.5-6.3L3 8M3 3v5h5M3 12a9 9 0 0 0 15.5 6.3L21 16M21 21v-5h-5" />
  </Icon>
);
export const IconSkip = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="10" />
    <path d="M8 12h8" />
  </Icon>
);
export const IconChevron = (p) => (
  <Icon {...p}>
    <path d="m6 9 6 6 6-6" />
  </Icon>
);
export const IconCloud = (p) => (
  <Icon {...p}>
    <path d="M17.5 19a4.5 4.5 0 1 0-1.4-8.8A6 6 0 1 0 6 18.5" />
    <path d="M6 19h11.5" />
  </Icon>
);
export const IconCpu = (p) => (
  <Icon {...p}>
    <rect x="5" y="5" width="14" height="14" rx="2" />
    <rect x="9" y="9" width="6" height="6" />
    <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
  </Icon>
);

/* ---------- Layout / feedback ---------- */

export function PageHeader({ title, description, children }) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
          {title}
        </h1>
        {description && (
          <p className="mt-1 max-w-2xl text-sm text-slate-600">{description}</p>
        )}
      </div>
      {children}
    </header>
  );
}

export function Alert({ children, tone = "error" }) {
  const tones = {
    error: "border-red-200 bg-red-50 text-red-800",
    info: "border-brand-200 bg-brand-50 text-brand-900",
  };
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={`flex items-start gap-2 rounded-lg border px-4 py-3 text-sm ${tones[tone]}`}
    >
      <IconAlert className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="min-w-0 break-words">{children}</div>
    </div>
  );
}

export function EmptyState({ title, children }) {
  return (
    <div className="card px-6 py-12 text-center">
      <p className="font-medium text-slate-900">{title}</p>
      <div className="mt-1 text-sm text-slate-600">{children}</div>
    </div>
  );
}

export function Skeleton({ className = "h-4 w-full" }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded bg-slate-200 ${className}`}
    />
  );
}

/* ---------- Status ---------- */

const STATUS = {
  queued: { cls: "bg-slate-100 text-slate-700 ring-slate-200", icon: IconClock },
  pending: { cls: "bg-slate-100 text-slate-700 ring-slate-200", icon: IconClock },
  running: { cls: "bg-blue-50 text-blue-800 ring-blue-200", icon: IconSpinner },
  completed: { cls: "bg-green-50 text-green-800 ring-green-200", icon: IconCheck },
  success: { cls: "bg-green-50 text-green-800 ring-green-200", icon: IconCheck },
  failed: { cls: "bg-red-50 text-red-800 ring-red-200", icon: IconX },
  skipped: { cls: "bg-slate-50 text-slate-600 ring-slate-300", icon: IconSkip },
};

export function statusStyle(status) {
  return STATUS[status] || STATUS.queued;
}

export function StatusBadge({ status }) {
  const { cls, icon: StatusIcon } = statusStyle(status);
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium capitalize ring-1 ring-inset ${cls}`}
    >
      <StatusIcon className="h-3.5 w-3.5" />
      {status}
    </span>
  );
}

/* ---------- Buttons ---------- */

const BTN = {
  primary:
    "bg-brand-600 text-white hover:bg-brand-700 active:bg-brand-800 shadow-sm",
  secondary:
    "bg-white text-slate-800 border border-slate-300 hover:bg-slate-50",
  danger: "text-red-700 hover:bg-red-50",
};

export function Button({
  variant = "primary",
  loading = false,
  className = "",
  children,
  disabled,
  ...props
}) {
  return (
    <button
      disabled={disabled || loading}
      className={`inline-flex min-h-[44px] items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50 ${BTN[variant]} ${className}`}
      {...props}
    >
      {loading && <IconSpinner />}
      {children}
    </button>
  );
}

/* ---------- Form controls (visible labels, helper text) ---------- */

function Field({ label, hint, id, children }) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-slate-800">
        {label}
      </label>
      {children}
      {hint && (
        <p id={`${id}-hint`} className="mt-1 text-xs text-slate-500">
          {hint}
        </p>
      )}
    </div>
  );
}

export function Input({ label, hint, ...props }) {
  const id = useId();
  return (
    <Field label={label} hint={hint} id={id}>
      <input
        id={id}
        aria-describedby={hint ? `${id}-hint` : undefined}
        className="field mt-1"
        {...props}
      />
    </Field>
  );
}

export function Textarea({ label, hint, ...props }) {
  const id = useId();
  return (
    <Field label={label} hint={hint} id={id}>
      <textarea
        id={id}
        aria-describedby={hint ? `${id}-hint` : undefined}
        rows={3}
        className="field mt-1 resize-y"
        {...props}
      />
    </Field>
  );
}

export function Select({ label, hint, children, ...props }) {
  const id = useId();
  return (
    <Field label={label} hint={hint} id={id}>
      <select
        id={id}
        aria-describedby={hint ? `${id}-hint` : undefined}
        className="field mt-1"
        {...props}
      >
        {children}
      </select>
    </Field>
  );
}

export function Checkbox({ label, ...props }) {
  return (
    <label className="flex min-h-[44px] items-center gap-3 text-sm text-slate-800">
      <input
        type="checkbox"
        className="h-5 w-5 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
        {...props}
      />
      {label}
    </label>
  );
}

export function FileInput({ label, hint, ...props }) {
  const id = useId();
  return (
    <Field label={label} hint={hint} id={id}>
      <input
        id={id}
        type="file"
        aria-describedby={hint ? `${id}-hint` : undefined}
        className="mt-1 block w-full text-sm text-slate-600 file:mr-4 file:rounded-lg file:border-0 file:bg-brand-50 file:px-4 file:py-2.5 file:text-sm file:font-medium file:text-brand-700 hover:file:bg-brand-100"
        {...props}
      />
    </Field>
  );
}
