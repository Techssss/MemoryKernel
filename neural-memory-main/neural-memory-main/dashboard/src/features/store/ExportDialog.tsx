import { useState } from "react"
import { Dialog, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Export, CheckCircle, Warning, Globe, ArrowRight } from "@phosphor-icons/react"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"
import { useExportBrain, usePublishBrain } from "@/api/hooks/useStore"

const CATEGORIES = [
  "general",
  "programming",
  "devops",
  "writing",
  "science",
  "security",
  "data",
  "design",
  "personal",
] as const

interface ExportDialogProps {
  open: boolean
  onClose: () => void
  brainName: string
}

export function ExportDialog({ open, onClose, brainName }: ExportDialogProps) {
  const { t } = useTranslation()
  const exportBrain = useExportBrain()
  const publishBrain = usePublishBrain()

  const [displayName, setDisplayName] = useState("")
  const [description, setDescription] = useState("")
  const [author, setAuthor] = useState("")
  const [category, setCategory] = useState<string>("general")
  const [tags, setTags] = useState("")
  const [step, setStep] = useState<"form" | "published">("form")
  const [publishResult, setPublishResult] = useState<{ pr_url: string; name: string } | null>(null)

  const formData = () => ({
    display_name: displayName.trim(),
    description: description.trim(),
    author: author.trim() || "anonymous",
    category,
    tags: tags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean),
  })

  const handleExport = () => {
    if (!displayName.trim()) return

    exportBrain.mutate(formData(), {
      onSuccess: (data) => {
        toast.success(t("store.exportSuccess", { name: displayName }))
        // Trigger download — extract the .brain package (not the API wrapper)
        const brainPackage = data.package ?? data
        const blob = new Blob([JSON.stringify(brainPackage, null, 2)], {
          type: "application/json",
        })
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `${displayName.replace(/\s+/g, "-").toLowerCase()}.brain`
        a.click()
        URL.revokeObjectURL(url)
      },
      onError: () => {
        toast.error(t("store.exportError"))
      },
    })
  }

  const handlePublish = () => {
    if (!displayName.trim() || !description.trim()) return

    publishBrain.mutate(formData(), {
      onSuccess: (data) => {
        toast.success(t("store.publishSuccess", { name: displayName }))
        setPublishResult({ pr_url: data.pr_url, name: data.name })
        setStep("published")
      },
      onError: (err) => {
        toast.error(err instanceof Error ? err.message : t("store.publishError"))
      },
    })
  }

  const handleClose = () => {
    if (!exportBrain.isPending && !publishBrain.isPending) {
      setStep("form")
      setPublishResult(null)
      onClose()
    }
  }

  return (
    <Dialog open={open} onClose={handleClose}>
      {step === "form" && (
        <>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Export className="size-5" aria-hidden="true" />
              {t("store.exportBrain")}
            </DialogTitle>
            <p className="text-sm text-muted-foreground">
              {t("store.exportDesc", { name: brainName })}
            </p>
          </DialogHeader>

          <div className="space-y-4">
            {/* Display Name */}
            <div className="space-y-1.5">
              <label htmlFor="export-name" className="text-sm font-medium">
                {t("store.displayName")} *
              </label>
              <input
                id="export-name"
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Python Best Practices"
                maxLength={100}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>

            {/* Description */}
            <div className="space-y-1.5">
              <label htmlFor="export-desc" className="text-sm font-medium">
                {t("store.description")} *
              </label>
              <textarea
                id="export-desc"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Curated best practices for Python development..."
                maxLength={1000}
                rows={3}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
              />
            </div>

            {/* Author + Category */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label htmlFor="export-author" className="text-sm font-medium">
                  {t("store.author")}
                </label>
                <input
                  id="export-author"
                  type="text"
                  value={author}
                  onChange={(e) => setAuthor(e.target.value)}
                  placeholder="anonymous"
                  maxLength={100}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>
              <div className="space-y-1.5">
                <label htmlFor="export-category" className="text-sm font-medium">
                  {t("store.category")}
                </label>
                <select
                  id="export-category"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 cursor-pointer"
                >
                  {CATEGORIES.map((cat) => (
                    <option key={cat} value={cat}>
                      {t(`store.category${cat.charAt(0).toUpperCase() + cat.slice(1)}`)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Tags */}
            <div className="space-y-1.5">
              <label htmlFor="export-tags" className="text-sm font-medium">
                {t("store.tags")}
              </label>
              <input
                id="export-tags"
                type="text"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder="python, coding, best-practices"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
              <p className="text-xs text-muted-foreground">{t("store.tagsHint")}</p>
            </div>

            {/* Security note */}
            <div className="flex items-start gap-2 rounded-lg border border-border bg-muted/50 p-3">
              <Warning className="mt-0.5 size-4 shrink-0 text-amber-500" aria-hidden="true" />
              <p className="text-xs text-foreground/70">{t("store.exportSecurityNote")}</p>
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-2 pt-2">
              <Button
                variant="outline"
                onClick={handleClose}
                disabled={exportBrain.isPending || publishBrain.isPending}
              >
                {t("common.cancel")}
              </Button>
              <Button
                variant="outline"
                onClick={handleExport}
                disabled={!displayName.trim() || !description.trim() || exportBrain.isPending || publishBrain.isPending}
                className="cursor-pointer"
              >
                {exportBrain.isPending ? (
                  t("store.exporting")
                ) : (
                  <span className="flex items-center gap-2">
                    <Export className="size-4" aria-hidden="true" />
                    {t("store.downloadFile")}
                  </span>
                )}
              </Button>
              <Button
                onClick={handlePublish}
                disabled={!displayName.trim() || !description.trim() || publishBrain.isPending || exportBrain.isPending}
                className="cursor-pointer"
              >
                {publishBrain.isPending ? (
                  <span className="inline-flex items-center gap-1.5">
                    <span className="size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    {t("store.publishing")}
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <Globe className="size-4" aria-hidden="true" />
                    {t("store.publishToCommunity")}
                  </span>
                )}
              </Button>
            </div>
          </div>
        </>
      )}

      {step === "published" && publishResult && (
        <div className="space-y-4 text-center py-4">
          <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-emerald-500/10">
            <CheckCircle className="size-7 text-emerald-500" weight="fill" aria-hidden="true" />
          </div>
          <div>
            <h3 className="font-display text-lg font-bold text-foreground">
              {t("store.publishedTitle")}
            </h3>
            <p className="mt-2 text-sm text-muted-foreground">
              {t("store.publishedDesc")}
            </p>
          </div>
          {publishResult.pr_url && (
            <a
              href={publishResult.pr_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-sm text-primary hover:underline"
            >
              {t("store.viewPR")}
              <ArrowRight className="size-3.5" aria-hidden="true" />
            </a>
          )}
          <div className="pt-2">
            <Button variant="outline" onClick={handleClose}>
              {t("store.done")}
            </Button>
          </div>
        </div>
      )}
    </Dialog>
  )
}
