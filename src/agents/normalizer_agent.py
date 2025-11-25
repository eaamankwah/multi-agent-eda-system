class JSONXMLNormalizerAgent(BaseAgent):
    """Transforms semi-structured data into flat tabular formats"""

    def __init__(self):
        super().__init__("JSON/XML Normalizer")

    def normalize_json(self, data: Union[str, Dict, List]) -> pd.DataFrame:
        """Convert JSON to flat DataFrame"""
        self.log("Normalizing JSON data...")

        try:
            if isinstance(data, str):
                data = json.loads(data)

            # Handle nested structures
            df = pd.json_normalize(data)
            self.log(f"JSON normalized: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            self.log(f"JSON normalization error: {e}")
            return pd.DataFrame()

    def normalize_xml(self, xml_string: str) -> pd.DataFrame:
        """Convert XML to flat DataFrame"""
        self.log("Normalizing XML data...")

        try:
            import xmltodict
            data_dict = xmltodict.parse(xml_string)
            df = pd.json_normalize(data_dict)
            self.log(f"XML normalized: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            self.log(f"XML normalization error: {e}")
            return pd.DataFrame()

    def auto_detect_and_normalize(self, data: Any) -> pd.DataFrame:
        """Automatically detect format and normalize"""
        self.log("Auto-detecting data format...")

        if isinstance(data, pd.DataFrame):
            self.log("Data is already a DataFrame")
            return data

        if isinstance(data, str):
            # Try JSON first
            try:
                return self.normalize_json(data)
            except:
                pass

            # Try XML
            try:
                return self.normalize_xml(data)
            except:
                pass

        # Try dict or list
        if isinstance(data, (dict, list)):
            return self.normalize_json(data)

        self.log("Could not detect format, returning empty DataFrame")
        return pd.DataFrame()

    def flatten_nested_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten any remaining nested structures"""
        self.log("Flattening nested columns...")

        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                try:
                    nested_df = pd.json_normalize(df[col].dropna())
                    nested_df.columns = [f"{col}.{subcol}" for subcol in nested_df.columns]
                    df = df.drop(columns=[col]).join(nested_df)
                except:
                    self.log(f"Could not flatten column: {col}")

        return df
