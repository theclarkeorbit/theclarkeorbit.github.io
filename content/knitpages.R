#!/usr/local/bin/Rscript --vanilla

# compiles all .Rmd files in _R directory into .md files in blog directory,
# if the input file is older than the output file.

# run ./knitpages.R to update all knitr files that need to be updated.
# run this script from your base content directory

# see https://www.r-bloggers.com/how-to-write-pelican-blog-posts-using-rmarkdown-knitr-2-0/

library(knitr)

KnitPost <- function(input, outfile, figsfolder, cachefolder, base.url="/") {
  opts_knit$set(base.url = base.url)
  fig.path <- paste0(figsfolder, sub(".Rmd$", "", basename(input)), "/")
  cache.path <- file.path(cachefolder, sub(".Rmd$", "", basename(input)), "/")

  opts_chunk$set(fig.path = fig.path)
  opts_chunk$set(cache.path = cache.path)
  opts_chunk$set(fig.cap = "center")
  opts_chunk$set(fig.width = 10.5)
  opts_chunk$set(fig.height = 7.5)
  render_markdown()
  knit(input, outfile, envir = parent.frame())
}

knit_folder <- function(infolder, outfolder, figsfolder, cachefolder, force = F) {
  for (infile in list.files(infolder, pattern = "*.Rmd", full.names = TRUE)) {

    print(infile)
    outfile = paste0(outfolder, "/", sub(".Rmd$", ".md", basename(infile)))
    print(outfile)

    # knit only if the input file is the last one modified
    if (!file.exists(outfile) | file.info(infile)$mtime > file.info(outfile)$mtime) {
        KnitPost(infile, outfile, figsfolder, cachefolder)
    }
  }
}

knit_folder("_R", "blogs/math_sci_tech", "figures/", "_caches")
