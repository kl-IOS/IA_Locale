-- filters.lua
-- 🔹 Supprime les préfixes "Slide X:" des titres
function Header(h)
  local txt = pandoc.utils.stringify(h.content)
  local new = txt:gsub("^Slide%s+%d+:%s*", "")
  if new ~= txt then
    h.content = { pandoc.Str(new) }
  end
  return h
end

-- 🔹 Convertit les blocs mermaid en <div class="mermaid">...</div>
function CodeBlock(cb)
  local classes = cb.classes or {}
  local has_mermaid = false
  for _, c in ipairs(classes) do
    if c == 'mermaid' or c == 'language-mermaid' then
      has_mermaid = true
      break
    end
  end
  if has_mermaid then
    local html = '<div class="mermaid">\n' .. cb.text .. '\n</div>'
    return pandoc.RawBlock('html', html)
  end
  return nil
end
