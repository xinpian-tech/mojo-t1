{ lib }:
{
  add =
    cache: seeds:
    lib.concatMapStrings (s: ''
      _d="${cache}/content_addressable/sha256/${s.sha256}"
      mkdir -p "$_d"
      cp --reflink=auto "${s.src}" "$_d/file"
      chmod 644 "$_d/file"
      : > "$_d/id-$(printf '%s' ${lib.escapeShellArg (lib.concatStringsSep " " s.urls)} | sha256sum | cut -d' ' -f1)"
    '') seeds;
  remove =
    cache: seeds:
    lib.concatMapStrings (s: ''
      rm -rf "${cache}/content_addressable/sha256/${s.sha256}"
    '') seeds;
}
