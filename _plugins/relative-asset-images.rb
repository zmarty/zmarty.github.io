# Keep post images readable in Markdown previews, where ../assets is relative
# to _posts. Published posts live at a different URL depth, so resolve these
# paths against the site's base URL before rendering.
Jekyll::Hooks.register :posts, :pre_render do |post|
  next unless post.data['relative_asset_images']

  baseurl = post.site.baseurl.to_s.sub(%r{/+\z}, '')
  post.content = post.content.gsub(/(<img\b[^>]*\bsrc=["'])\.\.\/assets\//i) do
    "#{Regexp.last_match(1)}#{baseurl}/assets/"
  end
end
