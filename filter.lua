
-- https://github.com/jgm/pandoc/issues/4894

function Image (img)
  if img.src:sub(1,1) == '/' then
    img.src = os.getenv 'WEBROOT' .. img.src
  end
  return img
end

function Link (link)
  if link.target:sub(1,1) == '/' then
    link.target = os.getenv 'WEBROOT' .. link.target
  end
  return link
end
