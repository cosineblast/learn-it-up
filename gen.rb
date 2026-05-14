#!/bin/ruby

# settings
WEBROOT = '/learn-it-up'

# execution
ENV['WEBROOT'] = WEBROOT

`mkdir -p build`
`cp *.jpg *.pdf build`

source_files = `ls *.md`.split("\n")

template = `cat template.html`

for source_file in source_files do

  basename = File.basename(source_file, File.extname(source_file))
  body_file = "build/#{basename}.body.html"
  target_file = "build/#{basename}.html"

  `pandoc --lua-filter="filter.lua" #{source_file} -o "#{body_file}"`

  result = (template.gsub '$STUFF', `cat "#{body_file}"`).gsub '$WEBROOT', WEBROOT

  File.write(target_file, result)

end
