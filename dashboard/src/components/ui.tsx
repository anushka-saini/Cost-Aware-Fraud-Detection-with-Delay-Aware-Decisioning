// Base components in the shadcn/ui style: small, unstyled-by-default pieces
// built on Radix primitives and class-variance-authority, owned by this repo.

import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as SliderPrimitive from '@radix-ui/react-slider'
import { cva, type VariantProps } from 'class-variance-authority'
import { X } from 'lucide-react'
import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type InputHTMLAttributes, type ReactNode,
  type SelectHTMLAttributes, type TdHTMLAttributes, type TextareaHTMLAttributes, type ThHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 whitespace-nowrap',
  {
    variants: {
      variant: {
        primary: 'bg-accent text-accent-contrast hover:opacity-90',
        outline: 'border bg-surface hover:bg-surface-2',
        ghost: 'hover:bg-surface-2',
        allow: 'border border-allow/40 bg-allow-soft text-allow-ink hover:border-allow',
        block: 'border border-block/40 bg-block-soft text-block-ink hover:border-block',
        review: 'border border-review/50 bg-review-soft text-review-ink hover:border-review',
      },
      size: { sm: 'h-8 px-2.5 text-[13px]', md: 'h-9 px-3.5', icon: 'h-8 w-8' },
    },
    defaultVariants: { variant: 'outline', size: 'md' },
  },
)

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, type = 'button', ...props }, ref) => (
  <button ref={ref} type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />
))
Button.displayName = 'Button'

export const Card = ({ className, ...props }: HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('rounded-lg border bg-surface', className)} {...props} />
)
export const CardHeader = ({ title, description, action, className }: { title: ReactNode; description?: ReactNode; action?: ReactNode; className?: string }) => (
  <div className={cn('flex items-start justify-between gap-4 px-4 pt-4', className)}>
    <div className="min-w-0">
      <h3 className="text-sm font-semibold">{title}</h3>
      {description && <p className="mt-0.5 text-[13px] text-secondary">{description}</p>}
    </div>
    {action && <div className="shrink-0">{action}</div>}
  </div>
)
export const CardBody = ({ className, ...props }: HTMLAttributes<HTMLDivElement>) => <div className={cn('p-4', className)} {...props} />

const fieldClass = 'h-9 w-full rounded-md border bg-surface px-3 text-sm placeholder:text-muted disabled:opacity-50'
export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(({ className, ...props }, ref) => (
  <input ref={ref} className={cn(fieldClass, className)} {...props} />
))
Input.displayName = 'Input'
export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(({ className, ...props }, ref) => (
  <select ref={ref} className={cn(fieldClass, 'pr-8', className)} {...props} />
))
Select.displayName = 'Select'
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...props }, ref) => (
  <textarea ref={ref} className={cn(fieldClass, 'h-auto min-h-20 py-2', className)} {...props} />
))
Textarea.displayName = 'Textarea'
export const Label = ({ className, ...props }: HTMLAttributes<HTMLLabelElement> & { htmlFor?: string }) => (
  <label className={cn('mb-1 block text-xs font-medium text-secondary', className)} {...props} />
)

export const Table = ({ className, ...props }: HTMLAttributes<HTMLTableElement>) => (
  <div className="w-full overflow-x-auto">
    <table className={cn('w-full border-collapse text-sm', className)} {...props} />
  </div>
)
export const Th = ({ className, ...props }: ThHTMLAttributes<HTMLTableCellElement>) => (
  <th className={cn('border-b px-3 py-2 text-left text-xs font-medium whitespace-nowrap text-muted', className)} {...props} />
)
export const Td = ({ className, ...props }: TdHTMLAttributes<HTMLTableCellElement>) => (
  <td className={cn('border-b px-3 py-2 align-middle', className)} {...props} />
)

export const Skeleton = ({ className }: { className?: string }) => (
  <div className={cn('animate-pulse rounded-md bg-surface-2', className)} aria-hidden="true" />
)

export const Kbd = ({ children }: { children: ReactNode }) => (
  <kbd className="rounded border bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-secondary">{children}</kbd>
)

export function Slider({ className, ...props }: SliderPrimitive.SliderProps & { 'aria-label': string }) {
  return (
    <SliderPrimitive.Root className={cn('relative flex h-5 w-full touch-none items-center select-none', className)} {...props}>
      <SliderPrimitive.Track className="relative h-1 w-full grow rounded-full bg-surface-2">
        <SliderPrimitive.Range className="absolute h-full rounded-full bg-accent" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb aria-label={props['aria-label']}
        className="block h-4 w-4 rounded-full border-2 border-accent bg-surface shadow-sm" />
    </SliderPrimitive.Root>
  )
}

export function Dialog({ open, onOpenChange, title, description, children }: {
  open: boolean; onOpenChange: (open: boolean) => void; title: string; description?: string; children: ReactNode
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <DialogPrimitive.Content
          className="fixed top-1/2 left-1/2 z-50 w-[min(92vw,30rem)] -translate-x-1/2 -translate-y-1/2 rounded-lg border bg-surface p-5 shadow-lg">
          <div className="flex items-start justify-between gap-4">
            <div>
              <DialogPrimitive.Title className="text-base font-semibold">{title}</DialogPrimitive.Title>
              {description && <DialogPrimitive.Description className="mt-1 text-[13px] text-secondary">{description}</DialogPrimitive.Description>}
            </div>
            <DialogPrimitive.Close asChild>
              <Button variant="ghost" size="icon" aria-label="Close"><X className="h-4 w-4" /></Button>
            </DialogPrimitive.Close>
          </div>
          <div className="mt-4">{children}</div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
