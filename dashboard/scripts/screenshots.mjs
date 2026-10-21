// Take a screenshot of every dashboard page into docs/screenshots/.
//
// Needs the API on port 8000 and the dashboard on port 5173, and an installed
// Chromium-based browser (Edge by default; set BROWSER_PATH to use another).
//
//     node scripts/screenshots.mjs
//
// It signs in as the admin, replays some of the live feed so the pages have
// data, makes a few review decisions, and captures each page.

import { mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'

const BASE = process.env.DASHBOARD_URL || 'http://localhost:5173'
const BROWSER = process.env.BROWSER_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../../docs/screenshots')
const STREAM_ROWS = Number(process.env.STREAM_ROWS || 700)
const wait = (ms) => new Promise((r) => setTimeout(r, ms))

mkdirSync(OUT, { recursive: true })
const browser = await puppeteer.launch({ executablePath: BROWSER, headless: true, defaultViewport: { width: 1440, height: 900 } })
const page = await browser.newPage()
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }])   // light by default; dark is toggled below
const errors = []
page.on('pageerror', (e) => errors.push(`page error: ${e.message}`))
page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`) })

const shot = async (name, fullPage = true) => {
  await wait(700)
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage })
  console.log('saved', `${name}.png`)
}
const go = async (path, selector) => {
  await page.goto(BASE + path, { waitUntil: 'networkidle0' })
  if (selector) await page.waitForSelector(selector, { timeout: 30000 })
}
const clickText = async (text, tag = 'button') => {
  const handle = await page.evaluateHandle((t, g) => [...document.querySelectorAll(g)].find((el) => el.textContent.trim().includes(t)), text, tag)
  const element = handle.asElement()
  if (!element) throw new Error(`No ${tag} containing "${text}"`)
  await element.click()
}
const api = (path, token, body) => page.evaluate(async (p, t, b) => {
  const r = await fetch('/api' + p, { method: b ? 'POST' : 'GET', headers: { Authorization: `Bearer ${t}`, 'Content-Type': 'application/json' }, body: b ? JSON.stringify(b) : undefined })
  return r.json()
}, path, token, body)

// 1. Login
await go('/')
await page.waitForSelector('#username')
await shot('01-login', false)
await clickText('Admin')
await clickText('Sign in')
await page.waitForSelector('nav[aria-label="Main"]')
const token = await page.evaluate(() => localStorage.getItem('fraud.token'))

// 2. Live feed: replay some transactions
await go('/live', 'select[aria-label="Replay speed"]')
await page.select('select[aria-label="Replay speed"]', '60')
await clickText('Restart')
await page.waitForFunction((n) => {
  const tile = [...document.querySelectorAll('div')].find((d) => d.textContent === 'Scored in this session')
  return tile && Number(tile.nextElementSibling.textContent.replace(/,/g, '')) >= n
}, { timeout: 600000, polling: 1000 }, STREAM_ROWS)
await clickText('Pause')
await shot('03-live-feed', false)
await clickText('Stop')

// A few decisions so the queue, audit log and similar cases have content.
const open = (await api('/cases?status=open&sort=risk&limit=50', token)).cases
for (const [i, c] of open.slice(0, 6).entries()) {
  const action = i % 3 === 2 ? 'approve' : i === 4 ? 'escalate' : 'decline'
  await api(`/cases/${c.id}/decision`, token, { action, note: action === 'decline' ? 'Recipient emptied the account immediately; customer confirmed they did not make it.' : action === 'approve' ? 'Known supplier payout, matches earlier pattern.' : 'Large amount and unclear recipient: needs a second opinion.' })
}

// 3. Pages
await go('/', 'h1')
await shot('02-overview')
await go('/queue', 'tbody tr')
await shot('04-review-queue', false)
const remaining = (await api('/cases?status=open&sort=risk&limit=5', token)).cases
await go(`/cases/${remaining[0].id}`, '[aria-label="Feature contributions"]')
await shot('05-case-detail')
await go('/simulator', '[aria-label="Review threshold"]')
await shot('06-threshold-simulator')
await clickText('Held-out')
await page.waitForSelector('[aria-label="Review threshold"]')
await shot('06b-threshold-simulator-held-out')
await go('/monitoring', 'table')
await shot('07-monitoring')
await go('/models', 'table')
await shot('08-model-comparison')
await go('/audit', 'tbody tr')
await shot('09-audit-log', false)

// Dark mode, two representative pages
await page.click('button[aria-label="Switch to dark mode"]')
await go('/simulator', '[aria-label="Review threshold"]')
await shot('10-dark-threshold-simulator', false)
await go(`/cases/${remaining[0].id}`, '[aria-label="Feature contributions"]')
await shot('11-dark-case-detail', false)
await page.click('button[aria-label="Switch to light mode"]')

// Analyst view: different navigation
await page.click('button[aria-label="Sign out"]')
await page.waitForSelector('#username')
await clickText('Analyst')
await clickText('Sign in')
await page.waitForSelector('nav[aria-label="Main"]')
await go('/queue', 'h1')
await shot('12-analyst-navigation', false)

await browser.close()
if (errors.length) {
  console.log(`\n${errors.length} browser errors:`)
  for (const e of [...new Set(errors)].slice(0, 20)) console.log(' -', e)
  process.exitCode = 1
} else console.log('\nNo browser errors.')
