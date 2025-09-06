"""Functions to handle all operations on the PDF's."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any

from pdfminer.layout import LTChar
from pdfminer.layout import LTImage
from pdfminer.layout import LTTextLineHorizontal
from pdfminer.layout import LTTextLineVertical
from pypdf import PdfReader
from pypdf import PdfWriter
from pypdf._utils import StrByteType

from .core import TableList
from .parsers import Hybrid
from .parsers import Lattice
from .parsers import Network
from .parsers import Stream
from .utils import TemporaryDirectory
from .utils import download_url
from .utils import get_image_char_and_text_objects
from .utils import get_page_layout
from .utils import get_rotation
from .utils import is_url


PARSERS = {
    "lattice": Lattice,
    "stream": Stream,
    "network": Network,
    "hybrid": Hybrid,
}


class PDFHandler:
    """Handles all operations on the PDF's.

    Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.

    Parameters
    ----------
    filepath : str
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.
    debug : bool, optional (default: False)
        Whether the parser should store debug information during parsing.
    """

    def __init__(
        self,
        filepath: StrByteType | Path | str,
        pages="1",
        password=None,
        debug=False,
        respect_permissions=False,
    ):
        self.debug = debug
        if is_url(filepath):
            filepath = download_url(str(filepath))
        self.filepath: StrByteType | Path | str = filepath

        if isinstance(filepath, str) and not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""  # noqa: S105
        else:
            self.password = password
        self.pages = self._get_pages(pages)
        self.respect_permissions = bool(respect_permissions)

    def _get_pages(self, pages):
        """Convert pages string to list of integers.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        pages : str, optional (default: '1')
            Comma-separated page numbers.
            Example: '1,3,4' or '1,4-end' or 'all'.

        Returns
        -------
        P : list
            List of int page numbers.

        """
        page_numbers = []

        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            infile = PdfReader(self.filepath, strict=False)

            if infile.is_encrypted:
                infile.decrypt(self.password)

            if pages == "all":
                page_numbers.append({"start": 1, "end": len(infile.pages)})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = len(infile.pages)
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})

        result = []
        for p in page_numbers:
            result.extend(range(p["start"], p["end"] + 1))
        return sorted(set(result))

    def _save_page(
        self, filepath: StrByteType | Path, page: int, temp: str, **layout_kwargs
    ) -> tuple[
        Any,
        tuple[float, float],
        list[LTImage],
        list[LTChar],
        list[LTTextLineHorizontal],
        list[LTTextLineVertical],
    ]:
        """Saves specified page from PDF into a temporary directory.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        page : int
            Page number.
        temp : str
            Tmp directory.


        Returns
        -------
        layout : object

        dimensions : tuple
            The dimensions of the pdf page

        filepath : str
            The path of the single page PDF - either the original, or a
            normalized version.

        """
        infile = PdfReader(filepath, strict=False)
        # Respect user permissions before splitting (deny if extraction not allowed)
        if getattr(infile, "is_encrypted", False):
            if self.respect_permissions:
                allowed = False

                # Try modern bitmask first
                try:
                    from pypdf.constants import UserAccessPermissions as UAP  # type: ignore
                except Exception:
                    UAP = None  # type: ignore

                try:
                    uap = getattr(infile, "user_access_permissions", None)
                    if (uap is not None) and (UAP is not None):
                        if (uap & getattr(UAP, "EXTRACT", 0)) or (
                            uap & getattr(UAP, "EXTRACT_TEXT_AND_GRAPHICS", 0)
                        ):
                            allowed = True
                except Exception:
                    pass

                # Fallback via decode_permissions on encryption dict
                if not allowed:
                    try:
                        enc = getattr(infile, "_encryption", None)
                        if (enc is not None) and hasattr(infile, "decode_permissions"):
                            perms = infile.decode_permissions(enc.P)
                            allowed = bool(
                                perms.get("extract")
                                or perms.get("extract_text_and_graphics")
                            )
                    except Exception:
                        pass

                if not allowed:
                    raise PermissionError(
                        "PDF forbids text/graphics extraction. "
                        "Provide a password with sufficient rights or call read_pdf(..., respect_permissions=False)."
                    )

            # Either permissions allowed or override requested -> decrypt (if password provided)
            infile.decrypt(self.password)  # noqa: S105

        fpath = os.path.join(temp, f"page-{page}.pdf")
        froot, fext = os.path.splitext(fpath)
        p = infile.pages[page - 1]
        outfile = PdfWriter()
        outfile.add_page(p)
        with open(fpath, "wb") as f:
            outfile.write(f)
        layout, dimensions = get_page_layout(fpath, **layout_kwargs)
        # fix rotated PDF
        images, chars, horizontal_text, vertical_text = get_image_char_and_text_objects(
            layout
        )
        rotation = get_rotation(chars, horizontal_text, vertical_text)
        if rotation:
            # Windows-friendly: use a temp file, then rewrite original once.
            with tempfile.NamedTemporaryFile(
                prefix="camelot-rot-", suffix=fext, dir=temp, delete=False
            ) as tmp:
                tmp_path = tmp.name
            try:
                with open(fpath, "rb") as src, open(tmp_path, "wb") as dst:
                    dst.write(src.read())
                with open(tmp_path, "rb") as instream:
                    infile = PdfReader(instream, strict=False)
                    if infile.is_encrypted:
                        infile.decrypt(self.password)
                    page0 = infile.pages[0]
                    if rotation == "anticlockwise":
                        page0.rotate(90)
                    elif rotation == "clockwise":
                        page0.rotate(-90)
                    out = PdfWriter()
                    out.add_page(page0)
                    with open(fpath, "wb") as outpdf:
                        out.write(outpdf)
            finally:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass
            # Recompute layout after rotation
            layout, dimensions = get_page_layout(fpath, **layout_kwargs)
            images, chars, horizontal_text, vertical_text = (
                get_image_char_and_text_objects(layout)
            )
            return layout, dimensions, images, chars, horizontal_text, vertical_text

    def parse(
        self,
        flavor: str = "lattice",
        suppress_stdout: bool = False,
        parallel: bool = False,
        workers: int | None = None,
        layout_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Extract tables by calling parser.get_tables on all single page PDFs.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use.
            Lattice is used by default.
        suppress_stdout : bool (default: False)
            Suppress logs and warnings.
        parallel : bool (default: False)
            Process pages in parallel using all available cpu cores.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams
            <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.
        kwargs : dict
            See camelot.read_pdf kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        if layout_kwargs is None:
            layout_kwargs = {}

        tables = []
        # parser = Lattice(**kwargs) if flavor == "lattice" else Stream(**kwargs)
        parser_obj = PARSERS[flavor]
        parser = parser_obj(debug=self.debug, **kwargs)
        # Per-page parsing (and any hybrid behavior) is handled below in _parse_page.
        # If flavor == "hybrid", the Hybrid parser implements lattice→stream fallback per page.
        with TemporaryDirectory() as tempdir:
            cpu_count = max(1, mp.cpu_count())
            max_workers = (
                cpu_count if workers is None else max(1, min(int(workers), cpu_count))
            )
            use_mp = parallel and len(self.pages) > 1 and max_workers > 1
            if use_mp:
                # cross-platform stable multiprocessing
                with mp.get_context("spawn").Pool(processes=max_workers) as pool:
                    jobs = []
                    for p in self.pages:
                        j = pool.apply_async(
                            self._parse_page,
                            (p, tempdir, parser, suppress_stdout, layout_kwargs),
                        )
                        jobs.append(j)

                    for j in jobs:
                        t = j.get()
                        tables.extend(t)
            else:
                for p in self.pages:
                    t = self._parse_page(
                        p, tempdir, parser, suppress_stdout, layout_kwargs
                    )
                    tables.extend(t)

    def _parse_page(
        self, page: int, tempdir: str, parser, suppress_stdout: bool, layout_kwargs
    ):
        """Extract tables by calling parser.get_tables on a single page PDF.

        Parameters
        ----------
        page : int
            Page number to parse
        parser : Lattice, Stream, Network or Hybrid
            The parser to use.
        suppress_stdout : bool
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams
            <https://pdfminersix.readthedocs.io/en/latest/reference/composable.html#laparams>`_ kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.

        """
        layout, dimensions, images, chars, horizontal_text, vertical_text = (
            self._save_page(self.filepath, page, tempdir, **layout_kwargs)
        )
        page_path = os.path.join(tempdir, f"page-{page}.pdf")
        parser.prepare_page_parse(
            page_path,
            layout,
            dimensions,
            page,
            images,
            horizontal_text,
            vertical_text,
            layout_kwargs=layout_kwargs,
        )
        tables = parser.extract_tables()
        return tables
