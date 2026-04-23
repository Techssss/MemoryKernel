import { useRef, useState, useDeferredValue } from "react"
import {
  Storefront,
  Package,
  UploadSimple,
  Export,
  ArrowRight,
  Brain,
  ShieldCheck,
  UsersThree,
} from "@phosphor-icons/react"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { useRegistry, useImportBrainFile } from "@/api/hooks/useStore"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { StoreFilters } from "./StoreFilters"
import { BrainCard } from "./BrainCard"
import { BrainPreviewDialog } from "./BrainPreviewDialog"
import { ExportDialog } from "./ExportDialog"

export default function StorePage() {
  const { t } = useTranslation()
  const [search, setSearch] = useState("")
  const [category, setCategory] = useState("")
  const [sortBy, setSortBy] = useState("created_at")
  const [previewName, setPreviewName] = useState<string | null>(null)
  const [showExport, setShowExport] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const importFile = useImportBrainFile()

  const deferredSearch = useDeferredValue(search)

  const handleFileImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    importFile.mutate(file, {
      onSuccess: (data) => {
        toast.success(t("store.importSuccess", { name: data.brain_name, neurons: data.neurons_imported }))
      },
      onError: (err) => {
        toast.error(err instanceof Error ? err.message : t("store.importError"))
      },
    })
    e.target.value = ""
  }

  const { data, isLoading, error, refetch } = useRegistry({
    category: category || undefined,
    search: deferredSearch || undefined,
    sort_by: sortBy,
  })

  const brains = data?.brains ?? []
  const showHero = !isLoading && !error && brains.length === 0 && !search && !category

  return (
    <div className="space-y-6 p-6">
      {/* Page Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-foreground inline-flex items-center gap-2.5">
            <Storefront className="size-7" aria-hidden="true" />
            {t("store.title")}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground max-w-lg">
            {t("store.subtitle")}
          </p>
        </div>
        <div className="flex gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".brain,.json"
            onChange={handleFileImport}
            className="hidden"
            aria-label={t("store.importFile")}
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={importFile.isPending}
            className="cursor-pointer"
          >
            <UploadSimple className="size-4 mr-1.5" aria-hidden="true" />
            {importFile.isPending ? t("store.importing") : t("store.importFile")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowExport(true)}
            className="cursor-pointer"
          >
            <Export className="size-4 mr-1.5" aria-hidden="true" />
            {t("store.exportMyBrain")}
          </Button>
        </div>
      </div>

      {/* Hero / How It Works — shown when registry is empty and no filters active */}
      {showHero && <StoreHero onImportClick={() => fileInputRef.current?.click()} onExportClick={() => setShowExport(true)} />}

      {/* Filters — always show unless hero is displayed */}
      {!showHero && (
        <StoreFilters
          search={search}
          onSearchChange={setSearch}
          category={category}
          onCategoryChange={setCategory}
          sortBy={sortBy}
          onSortChange={setSortBy}
        />
      )}

      {/* Content */}
      {isLoading && <StoreGridSkeleton />}

      {error && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <p className="text-sm text-destructive">{t("store.fetchError")}</p>
          <Button variant="outline" size="sm" onClick={() => refetch()} className="mt-4 cursor-pointer">
            {t("store.retry")}
          </Button>
        </div>
      )}

      {!isLoading && !error && brains.length === 0 && (search || category) && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Package className="size-12 text-muted-foreground/40" aria-hidden="true" />
          <p className="mt-3 text-sm text-muted-foreground">{t("store.emptySearch")}</p>
          <Button
            variant="outline"
            size="sm"
            className="mt-3 cursor-pointer"
            onClick={() => {
              setSearch("")
              setCategory("")
            }}
          >
            {t("store.clearFilters")}
          </Button>
        </div>
      )}

      {!isLoading && !error && brains.length > 0 && (
        <>
          <p className="text-xs text-muted-foreground">
            {t("store.showing", { count: brains.length })}
            {data?.cached && (
              <span className="ml-1.5 text-foreground/35">({t("store.cached")})</span>
            )}
          </p>
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {brains.map((manifest) => (
              <BrainCard
                key={manifest.id ?? manifest.name}
                manifest={manifest}
                onClick={() => setPreviewName(manifest.name)}
              />
            ))}
          </div>
        </>
      )}

      {/* Dialogs */}
      <BrainPreviewDialog
        brainName={previewName}
        onClose={() => setPreviewName(null)}
      />
      <ExportDialog
        open={showExport}
        onClose={() => setShowExport(false)}
        brainName="current"
      />
    </div>
  )
}

/* ── Hero / How It Works ────────────────────────────────── */

function StoreHero({
  onImportClick,
  onExportClick,
}: {
  onImportClick: () => void
  onExportClick: () => void
}) {
  const { t } = useTranslation()

  return (
    <div className="space-y-8">
      {/* Intro Banner */}
      <div className="rounded-xl border border-border bg-gradient-to-br from-primary/5 via-background to-primary/5 p-8 text-center">
        <Storefront className="mx-auto size-12 text-primary/60" aria-hidden="true" />
        <h2 className="mt-4 font-display text-xl font-bold text-foreground">
          {t("store.heroTitle")}
        </h2>
        <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
          {t("store.heroDesc")}
        </p>
      </div>

      {/* How It Works — 3 Steps */}
      <div>
        <h3 className="text-sm font-semibold text-foreground mb-4">{t("store.howItWorks")}</h3>
        <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
          <StepCard
            step={1}
            icon={<Brain className="size-6 text-primary" />}
            title={t("store.step1Title")}
            desc={t("store.step1Desc")}
          />
          <StepCard
            step={2}
            icon={<ShieldCheck className="size-6 text-emerald-500" />}
            title={t("store.step2Title")}
            desc={t("store.step2Desc")}
          />
          <StepCard
            step={3}
            icon={<UsersThree className="size-6 text-blue-500" />}
            title={t("store.step3Title")}
            desc={t("store.step3Desc")}
          />
        </div>
      </div>

      {/* CTA Buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
        <Button onClick={onExportClick} className="cursor-pointer">
          <Export className="size-4 mr-2" aria-hidden="true" />
          {t("store.ctaExport")}
          <ArrowRight className="size-4 ml-2" aria-hidden="true" />
        </Button>
        <Button variant="outline" onClick={onImportClick} className="cursor-pointer">
          <UploadSimple className="size-4 mr-2" aria-hidden="true" />
          {t("store.ctaImport")}
        </Button>
      </div>
    </div>
  )
}

function StepCard({
  step,
  icon,
  title,
  desc,
}: {
  step: number
  icon: React.ReactNode
  title: string
  desc: string
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-5 space-y-3">
      <div className="flex items-center gap-3">
        <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
          {step}
        </span>
        {icon}
      </div>
      <h4 className="font-display text-sm font-bold text-card-foreground">{title}</h4>
      <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
    </div>
  )
}

/* ── Skeleton ──────────────────────────────────────────── */

function StoreGridSkeleton() {
  return (
    <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="rounded-xl border border-border bg-card p-4 space-y-3">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-2/3" />
          <div className="flex gap-2 pt-2">
            <Skeleton className="h-5 w-12" />
            <Skeleton className="h-5 w-12" />
            <Skeleton className="h-5 w-12" />
          </div>
        </div>
      ))}
    </div>
  )
}
