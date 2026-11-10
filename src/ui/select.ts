import { h } from './dom.ts';
import { icon } from './icons.ts';

interface Item {
  value: string;
  label: string;
}

let openMenu: Menu | null = null;

/** A floating list anchored to a trigger: shared by the custom select and the symbol search. */
class Menu {
  readonly el = h('div', { class: 'pop', role: 'listbox' });
  private items: Item[] = [];
  private active = -1;
  private selected: string | null = null;
  private anchor: HTMLElement;
  private onPick: (item: Item) => void;
  isOpen = false;

  constructor(anchor: HTMLElement, onPick: (item: Item) => void) {
    this.anchor = anchor;
    this.onPick = onPick;
    this.el.addEventListener('mousedown', (e) => e.preventDefault()); // keep focus on the trigger
    this.el.addEventListener('click', (e) => {
      const opt = (e.target as HTMLElement).closest<HTMLElement>('[data-index]');
      if (opt) this.pick(Number(opt.dataset.index));
    });
    document.body.append(this.el);
  }

  setItems(items: Item[], selected: string | null): void {
    this.items = items;
    this.selected = selected;
    this.active = Math.max(0, items.findIndex((i) => i.value === selected));
    this.el.replaceChildren(
      ...(items.length
        ? items.map((it, i) =>
            h('div', { class: 'pop-item', role: 'option', 'data-index': i, 'aria-selected': String(it.value === selected) }, h('span', {}, it.label), it.value === selected ? icon('check', 15) : null),
          )
        : [h('div', { class: 'pop-empty' }, 'Sin resultados')]),
    );
    this.paintActive(false);
    if (this.isOpen) this.place();
  }

  private paintActive(scroll = true): void {
    this.el.querySelectorAll('.pop-item').forEach((n, i) => n.classList.toggle('active', i === this.active));
    if (scroll) this.el.querySelector('.pop-item.active')?.scrollIntoView({ block: 'nearest' });
  }

  private place(): void {
    const r = this.anchor.getBoundingClientRect();
    const below = innerHeight - r.bottom - 12;
    const above = r.top - 12;
    const up = below < 180 && above > below;
    const s = this.el.style;
    s.minWidth = `${r.width}px`;
    s.maxHeight = `${Math.min(300, Math.max(120, up ? above : below))}px`;
    s.left = `${Math.max(8, Math.min(r.left, innerWidth - this.el.offsetWidth - 8))}px`;
    s.top = up ? '' : `${r.bottom + 6}px`;
    s.bottom = up ? `${innerHeight - r.top + 6}px` : '';
    this.el.dataset.side = up ? 'top' : 'bottom';
  }

  open(): void {
    if (this.isOpen) return;
    openMenu?.close();
    openMenu = this;
    this.isOpen = true;
    this.place();
    this.el.classList.add('open');
    this.anchor.setAttribute('aria-expanded', 'true');
    this.paintActive();
  }

  close(): void {
    if (!this.isOpen) return;
    this.isOpen = false;
    if (openMenu === this) openMenu = null;
    this.el.classList.remove('open');
    this.anchor.setAttribute('aria-expanded', 'false');
  }

  move(delta: number): void {
    if (!this.items.length) return;
    this.active = (this.active + delta + this.items.length) % this.items.length;
    this.paintActive();
  }

  jump(prefix: string): void {
    const i = this.items.findIndex((it) => it.label.toLowerCase().startsWith(prefix));
    if (i >= 0) {
      this.active = i;
      this.paintActive();
    }
  }

  pick(index = this.active): boolean {
    const it = this.items[index];
    if (!it) return false;
    this.onPick(it);
    return true;
  }

  contains(target: Node): boolean {
    return this.el.contains(target) || this.anchor.contains(target);
  }

  get value(): string | null {
    return this.selected;
  }
}

document.addEventListener('pointerdown', (e) => {
  if (openMenu && !openMenu.contains(e.target as Node)) openMenu.close();
});
addEventListener('resize', () => openMenu?.close());
document.addEventListener(
  'scroll',
  (e) => {
    if (openMenu && !openMenu.el.contains(e.target as Node)) openMenu.close();
  },
  true,
);

/**
 * Replace a native <select> with a styled, animated listbox. The native element stays in the DOM
 * (hidden) as the source of truth: it still holds the value and fires `change`.
 * Returns a function to call after changing its options, value or disabled state from code.
 */
export function enhanceSelect(native: HTMLSelectElement): () => void {
  const label = h('span', { class: 'select-label' });
  const trigger = h('button', { type: 'button', class: 'select', 'aria-haspopup': 'listbox', 'aria-expanded': 'false' }, label, icon('chevronDown', 16));
  const menu = new Menu(trigger, (it) => {
    menu.close();
    if (native.value === it.value) return;
    native.value = it.value;
    native.dispatchEvent(new Event('change'));
    sync();
  });
  const sync = () => {
    const items = [...native.options].map((o) => ({ value: o.value, label: o.textContent ?? o.value }));
    menu.setItems(items, native.value);
    label.textContent = items.find((i) => i.value === native.value)?.label ?? '';
    trigger.disabled = native.disabled;
  };
  let typed = '';
  let typedAt = 0;
  trigger.addEventListener('click', () => (menu.isOpen ? menu.close() : menu.open()));
  trigger.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (menu.isOpen) menu.move(e.key === 'ArrowDown' ? 1 : -1);
      else menu.open();
    } else if ((e.key === 'Enter' || e.key === ' ') && menu.isOpen) {
      e.preventDefault();
      menu.pick();
    } else if (e.key === 'Escape' && menu.isOpen) {
      e.stopPropagation();
      menu.close();
    } else if (e.key === 'Tab') menu.close();
    else if (e.key.length === 1 && menu.isOpen) {
      typed = performance.now() - typedAt > 700 ? e.key : typed + e.key;
      typedAt = performance.now();
      menu.jump(typed.toLowerCase());
    }
  });
  trigger.addEventListener('blur', () => menu.close());
  if (native.id) {
    trigger.id = `${native.id}-btn`;
    document.querySelector(`label[for="${native.id}"]`)?.setAttribute('for', trigger.id);
  }
  native.hidden = true;
  native.tabIndex = -1;
  native.after(trigger);
  sync();
  return sync;
}

/** Search-as-you-type suggestions under a text input. Picking one fires `change` on the input. */
export function enhanceCombo(input: HTMLInputElement, getItems: () => string[]): void {
  const menu = new Menu(input, (it) => {
    menu.close();
    input.value = it.value;
    input.dispatchEvent(new Event('change'));
  });
  const refresh = () => {
    const q = input.value.trim().toUpperCase();
    const all = getItems();
    const starts = all.filter((s) => s.toUpperCase().startsWith(q));
    const rest = q ? all.filter((s) => !s.toUpperCase().startsWith(q) && s.toUpperCase().includes(q)) : [];
    const items = [...starts, ...rest].slice(0, 40).map((s) => ({ value: s, label: s }));
    menu.setItems(items, null);
    if (all.length) menu.open();
  };
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-expanded', 'false');
  input.addEventListener('focus', () => {
    input.select();
    refresh();
  });
  input.addEventListener('input', refresh);
  input.addEventListener('blur', () => menu.close());
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!menu.isOpen) refresh();
      else menu.move(e.key === 'ArrowDown' ? 1 : -1);
    } else if (e.key === 'Enter') {
      if (menu.isOpen && input.value.trim() && menu.pick()) e.preventDefault();
      else input.blur();
    } else if (e.key === 'Escape' && menu.isOpen) {
      e.stopPropagation();
      menu.close();
    }
  });
  // Called by the owner once the list has loaded while the input is focused.
  (input as HTMLInputElement & { refreshSuggestions?: () => void }).refreshSuggestions = () => {
    if (document.activeElement === input) refresh();
  };
}
