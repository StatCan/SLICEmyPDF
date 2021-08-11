import pandas as pd
import numpy as np
import re
import geopandas
import yaml
import delegator
import locale
import utilities.read_xml_json as pdx

from utilities.read_xml_json import auto_separate_tables
from lxml import etree
from PIL import Image, ImageDraw
from shapely.geometry import LineString, box
from scipy.signal import find_peaks
from scipy.stats import mode
from wand.image import Image as WImage
from wand.drawing import Drawing
from wand.color import Color

regex = re.compile(r"[^0-9$()]")
settings = yaml.safe_load(open(r"settings.yaml"))


class Extractor(object):
    """
    Class that encapsulates SLICE algorithm for information extraction
    from tabular data available in a PDF page.
    Object defined by path to pdf, page number, and few tuning parameters
    Private functions start with _
    ----------
    Parameters
    ----------
    pdf_loc : string (default: None)
        File path ending with .pdf
    page : int (default: 0)
        Page number with table to be extracted (default: 0)
    dist_threshold : int (default: 25)
        Split tokens having distance more than this value
    ver_prominence : int (default: None)
        Value of prominence as a parameter to scipy.signal.find_peaks()
    hor_prominence : int (default: None)
        Value of prominence as a parameter to scipy.signal.find_peaks()
    """

    def __init__(
                self,
                pdf_loc="",
                page=1,
                dist_threshold=25,
                ver_prominence=None,
                hor_prominence=None,
                **kwargs,
                ):
        self.pdf_loc = pdf_loc
        self.page = page
        self._validate_input()
        self.dist_threshold = dist_threshold
        self.ver_prominence = ver_prominence
        self.hor_prominence = hor_prominence

    def _validate_input(self):
        """
        Checks if input is in PDF format
        Raises exception with error message if input is not PDF
        ----------
        Input : Object
        Output : None
        """
        if self.pdf_loc != "":
            if self.pdf_loc.lower().endswith('.pdf'):
                pass
            else:
                raise Exception("Filename should end with .pdf")
        else:
            raise Exception("Provide path to .pdf file")

    def extract(self, FS_flag=True):
        """
        Run the table extraction pipeline
        ----------
        Input : Object, bool
            flag variable to indicate if PDF is a financial statement page
            with date columns
        Output : Pandas Dataframe
            Extracted table
        """
        coordinate_table, ver_list, hor_list, _, _ = self._get_token_coordinates(draw_img=False)
        _, extracted_table, date_table = self._convert_page_to_table(coordinate_table,
                                                                     ver_list,
                                                                     hor_list)
        if FS_flag:
            extracted_table = self._post_process_extracted_table(extracted_table,
                                                                 date_table)
        return extracted_table

    def get_pageview(self, save_img=False, file_name=""):
        """
        View page to be extracted in image format
        ----------
        Input : Object
        Output : Image
        """
        img = WImage(filename=self.pdf_loc+"["+str(self.page - 1)+"]", resolution=300)
        img.background_color = Color("white")
        img.alpha_channel = "remove"
        img.format = "jpeg"
        if save_img:
            self.save_image(img, file_name)
        return img

    def get_gridview(self, save_img=False, file_name=""):
        """
        View processed page with row and column divisions in image format
        ----------
        Input : Object
        Output : Image
        """
        _, _, _, img, _ = self._get_token_coordinates(draw_img=True)
        if save_img:
            self.save_image(img, file_name)
        return img

    def save_image(self, img, file_name=""):
        """
        Save image returned by get_* functions as JPEG
        ----------
        Input : Object
        Output : None
        """
        if file_name:
            if file_name.lower().endswith(('.jpg', '.jpeg')):
                img.save(filename=file_name)
            else:
                raise Exception("Filename in path should end with .jpg")
        else:
            img.save(filename=self.pdf_loc.replace('.pdf', '.jpg'))

    def _split_tokens(self, x):
        """
        Helper function
        Split tokens into separate bounding boxes based on whitespace between
        ----------
        Input : list
            word coordinates
        Output : list
            final coordinates for each bounding box
        """
        geo = [
            v[0].distance(v[1])
            for v in zip(x.word_geometry, x.word_geometry.shift(-1))
            if str(v[1]) != "nan"
        ]
        split = 0
        final_token = []
        for i in np.argwhere(np.array(geo) > self.dist_threshold).flatten("F"):
            df = pd.DataFrame([p.bounds for p in x.word_geometry.values[split: i + 1]])
            final_token.append(
                [
                    x["token_id"].values[split: i + 1],
                    df[0].min(),
                    df[1].min(),
                    df[2].max(),
                    df[3].max(),
                ]
            )
            split = i + 1
        df = pd.DataFrame([p.bounds for p in x.word_geometry.values[split:]])
        final_token.append(
            [
                x["token_id"].values[split:],
                df[0].min(),
                df[1].min(),
                df[2].max(),
                df[3].max(),
            ]
        )
        return final_token

    def _plot_image(self,
                    img,
                    table,
                    draw_type,
                    fill_opacity,
                    stroke_width,
                    stroke_color,
                    fill_color,
                    font_color,
                    stroke_dash_array=0,
                    layout="vertical",
                    delta=0,
                    ):
        """
        Helper function
        Draw rectangle bounding boxes on text in image
        Draw vertical and horizontal divisions
        ----------
        Input : multiple
            Image and other image parameters
        Output : Image
            Processed image
        """
        with Drawing() as draw:
            draw.fill_opacity = fill_opacity
            draw.stroke_width = stroke_width
            draw.stroke_color = Color(stroke_color)
            draw.fill_color = Color(fill_color)
            draw.font_color = Color(font_color)
            draw.stroke_dash_array = [stroke_dash_array]
            if draw_type == "rectangle":
                for idx, val in table.iterrows():
                    draw.rectangle(
                        val["word_xMin"],
                        val["word_yMin"],
                        val["word_xMax"],
                        val["word_yMax"],
                    )
                    draw.push()
            elif draw_type == "polyline":
                draw.polyline(
                    [(a - delta, b) for a, b in enumerate(table) if str(b) != "nan"]
                )
            elif draw_type == "line":
                if layout == "vertical":
                    for val in table:
                        draw.line((val - delta, 0), (val - delta, img.height))
                        draw.push()
                elif layout == "horizontal":
                    for val in table:
                        draw.line((0, val - delta), (img.width, val - delta))
                        draw.push()
            draw(img)
        return img

    def _plot_coordinates_on_image(self,
                                   img,
                                   coordinate_table,
                                   vertical_distance_list,
                                   horizontal_distance_list,
                                   ver_peaks,
                                   hor_peaks,
                                   b_w_img_numpy_vertical,
                                   b_w_img_numpy_horizontal,
                                   blue=False,
                                   purple=True,
                                   ):
        """
        Helper function
        Draw rectangle bounding boxes on text in image
        Draw vertical and horizontal divisions
        ----------
        Input : multiple
            Image and other SLICE parameters
        Output : Image
            Processed image
        """
        img = self._plot_image(
            img,
            table=coordinate_table,
            draw_type="rectangle",
            fill_opacity=0.1,
            stroke_width=3,
            stroke_color="red",
            fill_color="red",
            font_color="black",
        )

        if purple:
            img = self._plot_image(
                img,
                table=vertical_distance_list,
                draw_type="line",
                fill_opacity=0.1,
                stroke_width=2,
                stroke_color="purple",
                fill_color="purple",
                font_color="green",
                stroke_dash_array=10,
                delta=0,
            )
            img = self._plot_image(
                img,
                table=horizontal_distance_list,
                draw_type="line",
                fill_opacity=0.1,
                stroke_width=2,
                stroke_color="purple",
                fill_color="purple",
                font_color="green",
                stroke_dash_array=10,
                delta=0,
                layout="horizontal",
            )
        if blue:
            img = self._plot_image(
                img,
                table=ver_peaks,
                draw_type="line",
                fill_opacity=0.1,
                stroke_width=2,
                stroke_color="blue",
                fill_color="blue",
                font_color="green",
                stroke_dash_array=10,
                delta=20,
            )
            img = self._plot_image(
                img,
                table=hor_peaks,
                draw_type="line",
                fill_opacity=0.1,
                stroke_width=2,
                stroke_color="blue",
                fill_color="blue",
                font_color="green",
                stroke_dash_array=10,
                delta=10,
                layout="horizontal",
            )

        ver_img = WImage(
            width=len(b_w_img_numpy_vertical),
            height=int(b_w_img_numpy_vertical.fillna(0).max()),
            background=Color("white"),
        )
        ver_img = self._plot_image(
            ver_img,
            table=b_w_img_numpy_vertical,
            draw_type="polyline",
            fill_opacity=0.1,
            stroke_width=4,
            stroke_color="green",
            fill_color="green",
            font_color="green",
            delta=10,
        )
        if blue:
            ver_img = self._plot_image(
                ver_img,
                table=ver_peaks,
                draw_type="line",
                fill_opacity=0.1,
                stroke_width=4,
                stroke_color="black",
                fill_color="black",
                font_color="green",
                stroke_dash_array=20,
                delta=20,
            )
        if purple:
            ver_img = self._plot_image(
                ver_img,
                table=vertical_distance_list,
                draw_type="line",
                fill_opacity=0.1,
                stroke_width=4,
                stroke_color="black",
                fill_color="black",
                font_color="green",
                stroke_dash_array=20,
                delta=0,
            )

        ver_img.rotate(-180)
        ver_img.flop()
        ver_img.resize(len(b_w_img_numpy_vertical), 400)

        hor_img = WImage(
            width=len(b_w_img_numpy_horizontal),
            height=int(b_w_img_numpy_horizontal.fillna(0).max()),
            background=Color("white"),
        )
        hor_img = self._plot_image(
            hor_img,
            table=b_w_img_numpy_horizontal,
            draw_type="polyline",
            fill_opacity=0.1,
            stroke_width=4,
            stroke_color="green",
            fill_color="green",
            font_color="green",
            delta=10,
        )
        if blue:
            hor_img = self._plot_image(
                hor_img,
                table=hor_peaks,
                draw_type="line",
                fill_opacity=0.1,
                stroke_width=4,
                stroke_color="black",
                fill_color="black",
                font_color="green",
                stroke_dash_array=20,
                delta=10,
            )
        if purple:
            hor_img = self._plot_image(
                hor_img,
                table=horizontal_distance_list,
                draw_type="line",
                fill_opacity=0.1,
                stroke_width=4,
                stroke_color="black",
                fill_color="purple",
                font_color="green",
                stroke_dash_array=20,
                delta=0,
            )
        hor_img.resize(len(b_w_img_numpy_horizontal), 300)
        hor_img.rotate(-90)
        hor_img.flip()

        n_img = WImage(
            width=max(img.width, ver_img.width), height=img.height + ver_img.height
        )
        n_img.composite(image=img, left=0, top=0)
        n_img.composite(image=ver_img, left=0, top=img.height)

        f_img = WImage(
            width=hor_img.width + n_img.width,
            height=max(hor_img.height, n_img.height),
            background=Color("white"),
        )
        f_img.composite(image=hor_img, left=0, top=0)
        f_img.composite(image=n_img, left=hor_img.width, top=0)
        f_img.format = "jpeg"
        return img, f_img

    def _create_coordinate_table(self,
                                 pdf_text_path=settings["pdf_text_path"]):
        """Function to recursively parse the layout tree."""

        cmd = """{0} -bbox-layout -enc UTF-8 -f {1} -l {1} {2} -""".format(
            pdf_text_path, self.page, self.pdf_loc
        )
        a = delegator.run(cmd)
        return a.out

    def _create_coordinate_from_html_table(self,
                                           pdf_text_path=settings["pdf_html_path"]
                                           ):
        """Function to recursively parse the layout tree."""

        cmd = """{0} -xml -fontfullname -nodrm  -hidden  -i -f {1} -l {1} {2} output.xml""".format(
            pdf_text_path, self.page, self.pdf_loc
        )
        a = delegator.run(cmd)
        b = delegator.run("cat output.xml")
        xml_op = b.out
        b = delegator.run("rm output.xml")
        return xml_op

    def _get_date_position(self, x):
        """
        Helper function
        Post-processing start of table using date position
        """
        return [-1 * len(x) + i for v in x for i, item in enumerate(x) if item != ""]

    def _rolling_window(self, a, shape):
        """
        Helper function
        Rolling window for 2D array
        """
        s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
        strides = a.strides + a.strides
        val = np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)
        return val

    def _generate_box(self, b):
        """
        Helper function
        Generate bounding box
        """
        cor = (b[0][0] + "," + b[1][1]).split(",")
        b = box(float(cor[1]), float(cor[0]), float(cor[3]), float(cor[2]))
        return b

    def _get_date_columns(self, x, y, z):
        """
        Helper function
        """
        if len(z) > 0:
            return [y, x, z]

    def _process_number(self, x, idx, ind):
        """
        Helper function
        Process numerical values in financial statements
        """
        x = list(filter(re.compile(r"[^$]").match, x))
        try:
            if idx == 0:
                return x[ind]

            x = x[ind].replace(" ", ",")
            locale.setlocale(locale.LC_ALL, "")
            tbl3 = str.maketrans("", "", "$,)")
            return float(x.replace("(", "-").translate(tbl3))
        except Exception:
            return ""

    def _process_date_entries(self, columns, x):
        """
        Helper function
        Get start of table using year as indicator
        """
        ind = columns.loc[columns[1] == x.name, 2].iloc[0]
        op = [self._process_number(a, idx, ind) if len(a) > 0 else ""
              for idx, a in enumerate(x)]
        return op

    def _get_token_coordinates(self,
                               draw_img=False,
                               entry=pd.DataFrame()
                               ):
        """
        First function to run in extraction process
        Finds text bounding boxes and coordinates
        Finds horizontal and vertical page divisions
        --------
        Input : bool
            draw_img set to True to view page sections(rows and columns)
        Output : Pandas dataframe, list, list, Image, Image
            Coordinate_table gives information about token on a page,
            Vertical_distance_list gives location of vertical table sections
            Horizontal_distance_list gives location of horizontal table sections
            img is image of the sectioned page with histogram plot
            original is image of the sectioned page without histogram plot
        """
        try:
            # Get block,line and word-wise coordinate information from XML data
            xml_doc = self._create_coordinate_table()
            # Get word-wise coordinate information from HTML data
            # to cross-reference with XML extracted coordinates
            xml_doc_html = self._create_coordinate_from_html_table()

            parser = etree.XMLParser(recover=True)
            xml_doc = etree.fromstring(xml_doc, parser=parser)

            xml_doc_df = pdx.read_xml(
                etree.tostring(xml_doc, pretty_print=True).decode(
                    "utf-8", "backslashreplace"
                ),
                encoding="latin-1",
                transpose=True,
            )
            xml_doc_data = xml_doc_df.pipe(auto_separate_tables, [])

            xml_doc_html_df = pdx.read_xml(xml_doc_html, encoding="latin-1", transpose=True)
            xml_doc_html_data = xml_doc_html_df.pipe(auto_separate_tables, [])
            datapoints = xml_doc_html_data["text"]
            word_cols = list(
                set(datapoints.columns) - {"@font", "@height", "@left", "@top", "@width"}
            )
            datapoints["#text"] = (
                datapoints[word_cols].fillna("").apply(lambda x: "".join(x), axis=1)
            )
            datapoints = datapoints.merge(
                xml_doc_html_data["fontspec"].rename({"@id": "@font"}, axis=1), on="@font"
            )
            page_info = (
                xml_doc_html_data["pdf2xml"]
                .fillna("")
                .astype(str)
                .apply("".join, axis=0)
                .to_frame()
                .T
            )
            page_information = pd.concat(
                [
                    page_info.rename(
                        {
                            "@height": "page_height",
                            "@left": "page_left",
                            "@number": "page_number",
                            "@position": "page_position",
                            "@top": "page_top",
                            "@width": "page_width",
                        },
                        axis=1,
                    )
                ]
                * len(datapoints),
                ignore_index=True,
            )
            datapoints = pd.concat([datapoints, page_information], axis=1)
            datapoints.columns = [re.sub(r"@|#", "", v) for v in datapoints.columns]
            datapoints = datapoints[
                [
                    "top",
                    "left",
                    "width",
                    "height",
                    "font",
                    "text",
                    "size",
                    "family",
                    "color",
                    "page_height",
                    "page_left",
                    "page_number",
                    "page_position",
                    "page_top",
                    "page_width",
                ]
            ][datapoints.text != ""].reset_index(drop=True)

            int_cols = [
                "top",
                "left",
                "width",
                "height",
                "font",
                "size",
                "page_height",
                "page_left",
                "page_number",
                "page_top",
                "page_width",
            ]
            datapoints[int_cols] = datapoints[int_cols].apply(
                lambda x: x.astype("float").astype(int)
            )

            datapoints["word_xMax"] = datapoints["left"] + datapoints["width"]
            datapoints["word_yMax"] = datapoints["top"] + datapoints["height"]
            datapoints = (
                datapoints.reset_index(drop=True)
                .reset_index()
                .rename(
                    {"index": "word_id", "top": "word_yMin", "left": "word_xMin"}, axis=1
                )
            )

            coordinate_table = xml_doc_data["doc"]
            coordinate_table.columns = [
                re.sub(r"@|#", "", "_".join(v.split("|")[-2:]))
                for v in coordinate_table.columns
            ]
            coordinate_table[coordinate_table.columns[:-1]] = coordinate_table[
                coordinate_table.columns[:-1]
            ].apply(lambda x: x.astype("float"))
            coordinate_table = (
                coordinate_table.reset_index(drop=True)
                .reset_index()
                .rename({"index": "token_id"}, axis=1)
            )

            img = WImage(filename=self.pdf_loc + "[" + str(self.page - 1) + "]", resolution=300)
            img.background_color = Color("white")
            img.alpha_channel = "remove"
            w = int(datapoints.iloc[0].page_width)
            h = int(datapoints.iloc[0].page_height)
            new_w = img.width / w
            new_h = img.height / h

            w_token = int(coordinate_table.iloc[0].page_width)
            h_token = int(coordinate_table.iloc[0].page_height)
            new_w_token = img.width / w_token
            new_h_token = img.height / h_token

            datapoints["geometry"] = datapoints.apply(
                lambda val: box(
                    new_w * val["word_xMin"],
                    new_h * val["word_yMin"],
                    new_w * val["word_xMax"],
                    new_h * val["word_yMax"],
                ),
                axis=1,
            )
            datapoints["token_geometry"] = datapoints.apply(
                lambda val: box(
                    new_w * val["word_xMin"],
                    new_h * val["word_yMin"],
                    new_w * val["word_xMax"],
                    new_h * val["word_yMax"],
                ),
                axis=1,
            )
            coordinate_table["geometry"] = coordinate_table.apply(
                lambda val: box(
                    new_w_token * val["word_xMin"],
                    new_h_token * val["word_yMin"],
                    new_w_token * val["word_xMax"],
                    new_h_token * val["word_yMax"],
                ),
                axis=1,
            )
            coordinate_table["word_geometry"] = coordinate_table.apply(
                lambda val: box(
                    new_w_token * val["word_xMin"],
                    new_h_token * val["word_yMin"],
                    new_w_token * val["word_xMax"],
                    new_h_token * val["word_yMax"],
                ),
                axis=1,
            )

            coordinate_table = coordinate_table[
                ~coordinate_table.word_text.str.match(r"^(\s+|)\$(\s+|)$")
            ].reset_index(drop=True)

            datapoints_gdf = geopandas.GeoDataFrame(datapoints)
            coordinate_table_gdf = geopandas.GeoDataFrame(coordinate_table)
            coordinate_intersection = geopandas.overlay(
                datapoints_gdf, coordinate_table_gdf, how="intersection"
            )
            coordinate_intersection["overlap"] = (
                coordinate_intersection.geometry.area
                / coordinate_intersection.word_geometry.apply(lambda x: x.area)
                * 100
            )

            coordinate_intersection = coordinate_intersection[
                ["token_id", "word_id", "text", "word_text", "overlap", "word_geometry"]
            ]

            # Split tokens based on (whitespace) distance
            coordinate_intersection = coordinate_intersection[
                coordinate_intersection.overlap > 50
            ].reset_index(drop=True)
            coordinate_table = (
                coordinate_intersection.groupby(["word_id", "text"])
                .apply(lambda x: self._split_tokens(x))
                .reset_index()
            )

            coordinate_table = pd.DataFrame(
                coordinate_table.explode(0)[0].to_list(),
                columns=["token", "word_xMin", "word_yMin", "word_xMax", "word_yMax"],
            )
            coordinate_table["full_text"] = coordinate_table.token.apply(
                lambda x: [
                    coordinate_intersection[
                        coordinate_intersection.token_id == y
                    ].word_text.iloc[0]
                    for y in x
                ]
            )
            coordinate_table["page_width"] = img.width
            coordinate_table["page_height"] = img.height

            coordinate_table["date"] = coordinate_table.full_text.apply(
                lambda x: " ".join(x).lower()
            ).str.extract(".*([2][0-9]{3})")
            coordinate_table["date_other"] = coordinate_table.full_text.apply(
                lambda x: " ".join(x).lower()
            ).str.extract("([-][0-9][0-9])")
            coordinate_table["units"] = coordinate_table.full_text.apply(
                lambda x: " ".join(x).lower()
            ).str.extract(
                "("
                + "|".join([a.replace("$", r"\$").lower() for a in settings["units"]])
                + ")"
            )
            coordinate_table["currency"] = coordinate_table.full_text.apply(
                lambda x: " ".join(x).lower()
            ).str.extract(
                "("
                + "|".join([a.replace("$", r"\$").lower() for a in settings["currencies"]])
                + ")"
            )

            coordinate_table.loc[
                :, ["date", "date_other", "units", "currency"]
            ] = coordinate_table.loc[:, ["date", "date_other", "units", "currency"]].fillna(
                ""
            )
            coordinate_table.loc[:, "date"] = (
                coordinate_table["date"] + coordinate_table["date_other"]
            )

            # create an image
            b_w_img = Image.new("RGB", (img.width, img.height),
                                (255, 255, 255))
            draw_b_w_img = ImageDraw.Draw(b_w_img)

            for idx, val in coordinate_table.iterrows():
                diff = (val["word_yMax"] - val["word_yMin"]) * 0.25
                draw_b_w_img.rectangle(
                    [
                        val["word_xMin"],
                        val["word_yMin"] + diff,
                        val["word_xMax"],
                        val["word_yMax"] - diff,
                    ],
                    fill="black",
                    outline="white",
                    width=1,
                )
            # converting image to bilevel- only 0 or 1 values- using mode ='1'
            b_w_img_numpy_vertical = np.logical_not(np.array(b_w_img.convert("1"))).sum(0)
            b_w_img_numpy_vertical = (
                pd.Series(b_w_img_numpy_vertical).rolling(window=60).median()
            )

            if self.ver_prominence is None:
                ver_peaks, prominences = find_peaks(
                    b_w_img_numpy_vertical,
                    prominence=np.percentile(b_w_img_numpy_vertical.fillna(0), 45),
                )

            else:
                ver_peaks, prominences = find_peaks(
                    b_w_img_numpy_vertical,
                    prominence=self.ver_prominence
                )

            start = 0
            vertical_distance_list = []
            for end in ver_peaks:
                vertical_distance_list.append(
                    b_w_img_numpy_vertical[start:end].fillna(0).idxmin()
                )
                start = end + 1
            vertical_distance_list.append(b_w_img_numpy_vertical[start:].fillna(0).idxmin())
            vertical_distance_list.append(img.width)

            verz = [
                LineString([(val - 20, 0), (val - 20, img.height)]) for val in ver_peaks
            ]
            # converting image to greyscale using mode = 'L'
            b_w_img_numpy_horizontal = np.logical_not(np.array(b_w_img.convert("L"))).sum(1)
            b_w_img_numpy_horizontal_new = (
                pd.Series(b_w_img_numpy_horizontal).rolling(window=20).median()
            )
            b_w_img_numpy_horizontal = (
                pd.Series(b_w_img_numpy_horizontal)
                .rolling(window=10)
                .apply(lambda x: mode(x)[0])
            )
            hor_peaks, _ = find_peaks(b_w_img_numpy_horizontal)

            if self.hor_prominence is None:
                hor_peaks_new, prominences = find_peaks(
                    b_w_img_numpy_horizontal_new,
                    prominence=b_w_img_numpy_horizontal_new.fillna(0).median(),
                )

            else:
                hor_peaks_new, prominences = find_peaks(
                    b_w_img_numpy_horizontal_new,
                    prominence=self.hor_prominence
                )

            horz = [LineString([(0, val - 10), (img.width, val - 10)]) for val in hor_peaks]

            start = 0
            horizontal_distance_list = []
            for end in hor_peaks:
                horizontal_distance_list.append(
                    b_w_img_numpy_horizontal_new[start:end].fillna(0).idxmin()
                )
                start = end + 1
            horizontal_distance_list.append(
                b_w_img_numpy_horizontal_new[start:].fillna(0).idxmin()
            )
            horizontal_distance_list.append(img.height)

            if draw_img:
                original, img = self._plot_coordinates_on_image(
                    img,
                    coordinate_table,
                    vertical_distance_list,
                    horizontal_distance_list,
                    ver_peaks,
                    hor_peaks,
                    b_w_img_numpy_vertical,
                    b_w_img_numpy_horizontal,
                )

            else:
                img = original = ""
        except Exception:
            coordinate_table = pd.DataFrame()
            vertical_distance_list = horizontal_distance_list = []
            img = original = ""
            raise Exception("Unable to locate coordinates for text! Provide a valid path to a text-based PDF with a single table")
        return coordinate_table, vertical_distance_list,\
            horizontal_distance_list, img, original

    def _convert_page_to_table(self,
                               coordinate_table,
                               vertical_distance_list,
                               horizontal_distance_list
                               ):
        """
        Second function to run in extraction process
        Assigns text to page sections
        Gets start of table position using date as indicator
        ---------
        Input : Pandas dataframe, list , list
            Partial output of _get_token_coordinates fed as input
        Output : geopandas, pandas df, pandas df
            coordinate_intersection gives text mapped to each section
            extracted_table contains all text in tabular format
                            w/o post-processing
            date_table gives location of dates in the table
        """
        try:
            coordinate_table = (
                coordinate_table.reset_index(drop=True)
                .reset_index()
                .rename({"index": "token_id"}, axis=1)
            )
            coordinate_table["geometry"] = coordinate_table[
                "word_area"
            ] = coordinate_table.apply(
                lambda val: box(
                    val["word_xMin"], val["word_yMin"], val["word_xMax"], val["word_yMax"]
                ),
                axis=1,
            )
            coordinate_table = geopandas.GeoDataFrame(coordinate_table)

            arr = np.array(
                [
                    [",".join([str(a), str(b)]) for b in vertical_distance_list]
                    for a in horizontal_distance_list
                ]
            )
            val = self._rolling_window(arr, (2, 2))
            layout = pd.DataFrame(
                [
                    [h_idx, v_idx, self._generate_box(b)]
                    for h_idx, a in enumerate(val)
                    for v_idx, b in enumerate(a)
                ],
                columns=["horizontal_index", "vertical_index", "geometry"],
            )
            layout["box_coordinates"] = layout["geometry"]
            layout = geopandas.GeoDataFrame(layout)

            coordinate_intersection = geopandas.overlay(
                layout, coordinate_table, how="intersection"
            )
            coordinate_intersection["overlap"] = (
                coordinate_intersection.geometry.area
                / coordinate_intersection.word_area.apply(lambda x: x.area)
                * 100
            )
            table = (
                coordinate_intersection.sort_values("overlap")
                .drop_duplicates(subset=["token_id"], keep="last")
                .sort_values(["horizontal_index", "vertical_index", "word_xMin"])
            )

            extracted_table = (
                pd.DataFrame(
                    np.zeros((table.horizontal_index.max() + 1, table.vertical_index.max() + 1))
                )
                .replace(0, np.nan)
                .applymap(lambda x: [])
            )
            _table = (
                pd.DataFrame(
                    np.zeros((table.horizontal_index.max() + 1, table.vertical_index.max() + 1))
                )
                .replace(0, np.nan)
                .applymap(lambda x: [])
            )
            for idx, val in table.iterrows():
                extracted_table.iloc[val.horizontal_index, val.vertical_index].append(
                    [" ".join(val.full_text), val.date]
                )
            date_table = extracted_table.applymap(
                lambda x: self._get_date_position([v[1] for v in x]) if (len(x) > 0) else []
            )
            extracted_table = extracted_table.applymap(
                lambda x: [v[0] for v in x] if (len(x) > 0) else []
            )
        except Exception:
            coordinate_intersection = pd.DataFrame()
            extracted_table = pd.DataFrame()
            date_table = pd.DataFrame()
            raise Exception("Unable to map token to page sections! Please\
                            provide text-based PDF with a single table")
        return coordinate_intersection, extracted_table, date_table

    def _post_process_extracted_table(self, extracted_table, date_table):
        """
        Third and final function to run in extraction process
        Post-process table entries
        Should be modified as per requirement
        ---------
        Input : pandas df, pandas df
            Partial output of _convert_page_to_table fed as input
        Output : pandas df
            final_table is the extracted table with rule-based post-processing
        """
        try:
            date_columns = (
                date_table.apply(
                    lambda x: pd.DataFrame(x).apply(
                        lambda y: self._get_date_columns(
                            x.name, y.name, date_table.iloc[y.name, x.name]
                        ),
                        axis=1,
                    )
                )
                .fillna(np.nan)
                .values.flatten()
            )
            date_columns = [i for i in date_columns[~pd.isnull(date_columns)]]
            columns = pd.DataFrame(date_columns).explode(2)
            columns = columns[columns[1] > 0].reset_index(drop=True)
            unique_row = columns[0].value_counts().index[0]
            for c, d in columns[columns[0] != unique_row].iterrows():
                if abs(d[0] - unique_row) <= 2:
                    columns.loc[c, 0] = unique_row
                    extracted_table.loc[unique_row, d[1]] = extracted_table.loc[d[0], d[1]]
            columns = (
                columns[columns[0] == unique_row]
                .drop_duplicates(subset=[1], keep="first")
                .reset_index(drop=True)
            )
            date_columns_table = extracted_table.loc[unique_row:, columns[1].values]

            date_columns_table = date_columns_table.apply(
                lambda x: self._process_date_entries(columns, x)
            )

            o_table = extracted_table.loc[
                unique_row:, ~(extracted_table.columns.isin(columns[1]))
            ].applymap(lambda x: " ".join(x))

            final_table = pd.concat([o_table[0], date_columns_table], axis=1).reset_index(
                drop=True
            )
            # Combine lower row if word starts with lowercase
            upper_word = ""
            word_idx = 0
            mapping = {0: []}
            for idx, word_list in enumerate(final_table[0]):
                word = " ".join(word_list)
                if word != "" and word.strip()[0].islower() and idx != 0 and upper_word != "":
                    mapping[word_idx].append(idx)
                else:
                    upper_word = word
                    word_idx = idx
                    mapping[word_idx] = []

            for k, v in mapping.items():
                for i in v:
                    final_table.loc[i, 0] = " " + final_table.loc[i, 0]
                    final_table.loc[k, :] = final_table.loc[[k, i], :].astype(str).values.sum(0)
            final_table = final_table.loc[list(mapping.keys()), :]
            # Find rows with numerical values only
            # Assign statement to first column if it signifies total of a section
            labels = final_table[
                (~final_table[0].replace("", np.nan).isnull())
                & (
                    final_table[date_columns_table.columns]
                    .replace("", np.nan)
                    .isnull()
                    .all(axis=1)
                )
            ]
            values = final_table[
                (final_table[0].replace("", np.nan).isnull())
                & (
                    ~final_table[date_columns_table.columns]
                    .replace("", np.nan)
                    .isnull()
                    .all(axis=1)
                )
            ]
            values_new = final_table[
                ~(
                    (final_table[0].replace("", np.nan).isnull())
                    & (
                        ~final_table[date_columns_table.columns]
                        .replace("", np.nan)
                        .isnull()
                        .all(axis=1)
                    )
                )
            ]
            t = values_new.loc[1:, date_columns_table.columns].replace("", 0)
            v = final_table.loc[1:, date_columns_table.columns].replace("", 0)
            values_index = values.index.values
            labels_index = labels.index.values
            for i in values_index:
                if (i != 0) & (len(labels_index[labels_index < i]) > 0):
                    floor_val = labels_index[labels_index < i].max()
                    if (
                        t.loc[floor_val: i - 1].astype(float).sum(0) == v.loc[i]
                    ).all() is True:
                        final_table.loc[i, 0] = "Total " + \
                            final_table.loc[floor_val, 0]
                        labels_index = np.setdiff1d(labels_index, [floor_val])
        except Exception:
            final_table = extracted_table
            raise Exception("Unable to post-process table! Try extract(FS_flag=False)")
        return final_table