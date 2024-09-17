import pandas as pd
import numpy as np
from mooncloud_driver import abstract_probe, atom, result, entrypoint
from git_ci import gitCI
import gitlab
import github
from github import GithubException
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import typing
import io
import os

class MostImportantVarProbe(abstract_probe.AbstractProbe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.git_ci = None

    def requires_credential(self) -> any:
        return True

    def parse_input(self):
        config = self.config.input.get("config", {})
        self.host = config.get('target')
        self.repo_type = config.get('repo_type', '').lower()
        self.project = config.get('project')
        self.branch = config.get('branch', 'master')
        self.artifact_path = config.get('artifact_path')
        self.job_name = config.get('job_name') if self.repo_type == "gitlab" else None
        self.artifact_name = config.get('artifact_name') if self.repo_type == "github" else None

        if not self.host or not self.repo_type or not self.project or not self.artifact_path:
            raise ValueError("Missing required input fields")

    def setup_git_ci(self):
        token = self.config.credential.get('token')
        if not token:
            raise ValueError("Token not found in credentials")
        if self.repo_type == "gitlab":
            self.git_ci = gitCI(ci_type=gitCI.CIType.GITLAB, gl_domain=self.host, gl_token=token, gl_project=self.project)
        elif self.repo_type == "github":
            self.git_ci = gitCI(ci_type=gitCI.CIType.GITHUB, gh_domain=self.host, gh_token=token, gh_repo=self.project)
        else:
            raise ValueError("Unsupported repository type")

    def load_and_prepare_dataset(self):
        self.setup_git_ci()
        artifact_file_path = self.git_ci.getArtifact(branch_name=self.branch, job_name=self.job_name, artifact_path=self.artifact_path, artifact_name=self.artifact_name)

        if isinstance(artifact_file_path, io.TextIOWrapper):
            artifact_content = artifact_file_path.buffer.read()  
            artifact_file_path = f"/tmp/{os.path.basename(self.artifact_path)}"
            with open(artifact_file_path, 'wb') as f:
                f.write(artifact_content)
        elif not isinstance(artifact_file_path, str):
            raise ValueError("Unexpected artifact file type")

        if not os.path.exists(artifact_file_path):
            raise ValueError("Failed to download or find artifact")

        df = pd.read_csv(artifact_file_path)
        return df

    def filter_low_variance(self, df, threshold=0.1):
        variances = df.var()
        return df.loc[:, variances > threshold]

    def calculate_mi(self, col1, col2, df, numeric_cols):
        if col1 in numeric_cols or col2 in numeric_cols:
            mi = mutual_info_regression(df[[col1]].copy(), df[col2].copy())
        else:
            mi = mutual_info_classif(df[[col1]].copy(), df[col2].copy())
        return mi[0]

    def calculate_mutual_info_parallel(self, df, numeric_cols):
        mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
        results = Parallel(n_jobs=-1)(
            delayed(self.calculate_mi)(col1, col2, df, numeric_cols) for col1 in df.columns for col2 in df.columns
        )
        for idx, (col1, col2) in enumerate([(col1, col2) for col1 in df.columns for col2 in df.columns]):
            mi_matrix.at[col1, col2] = results[idx]
        return mi_matrix

    def evaluate_feature_importance(self, df):
        X = df
        y = np.random.choice([0, 1], size=(df.shape[0],)) 
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        impurity_importances = model.feature_importances_
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        permutation_importances = result.importances_mean
        importance = pd.DataFrame({
            'Feature': X.columns,
            'ImpurityImportance': impurity_importances,
            'PermutationImportance': permutation_importances
        }).sort_values(by='PermutationImportance', ascending=False)
        return importance

    def run_analysis(self, inputs: any) -> bool:
        df = self.load_and_prepare_dataset()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=[object, 'category']).columns.tolist()

        data_filtered = self.filter_low_variance(df[numeric_cols])
        data_encoded = pd.get_dummies(data_filtered, columns=categorical_cols, drop_first=True)

        scaler = StandardScaler()
        data_normalized = pd.DataFrame(scaler.fit_transform(data_encoded), columns=data_encoded.columns)

        mi_matrix = self.calculate_mutual_info_parallel(data_normalized, numeric_cols)
        mi_matrix = mi_matrix.apply(pd.to_numeric, errors='coerce')

        corr_matrix = data_normalized.corr()
        mi_threshold = np.nanpercentile(mi_matrix.values, 99.5)
        corr_threshold = 0.95

        sensitive_mi_attributes = mi_matrix[mi_matrix > mi_threshold].dropna(how='all').index.tolist()
        sensitive_corr_attributes = corr_matrix[(corr_matrix.abs() > corr_threshold) & (corr_matrix != 1.0)].dropna(how='all').index.tolist()

        potentially_sensitive_attributes = list(set(sensitive_mi_attributes) & set(sensitive_corr_attributes))

        importance = self.evaluate_feature_importance(data_normalized)

        important_sensitive_attributes = [attr for attr in potentially_sensitive_attributes if attr in importance.head(20)['Feature'].tolist()]

        result_data = {
            "potentially_sensitive_attributes": potentially_sensitive_attributes,
            "important_sensitive_attributes": important_sensitive_attributes
        }

        self.result.put_extra_data("analysis_result", result_data)
        
        if important_sensitive_attributes:
            self.result.integer_result = result.INTEGER_RESULT_FALSE
            self.result.pretty_result = f"Colonne potenzialmente sensibili e importanti: {important_sensitive_attributes}"
        else:
            self.result.integer_result = result.INTEGER_RESULT_TRUE
            self.result.pretty_result = "Nessuna delle colonne potenzialmente sensibili risulta importante."

        return True

    def atoms(self) -> typing.Sequence[atom.AtomPairWithException]:
        return [
            atom.AtomPairWithException(
                forward=self.parse_input,
                forward_captured_exceptions=[
                    atom.PunctualExceptionInformationForward(
                        exception_class=ValueError,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_parse_exception
                    )
                ]
            ),
            atom.AtomPairWithException(
                forward=self.load_and_prepare_dataset,
                forward_captured_exceptions=[
                    atom.PunctualExceptionInformationForward(
                        exception_class=ValueError,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_artifact_exception
                    ),
                    atom.PunctualExceptionInformationForward(
                        exception_class=gitlab.GitlabAuthenticationError,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_gitlab_auth_error
                    ),
                    atom.PunctualExceptionInformationForward(
                        exception_class=gitlab.GitlabGetError,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_gitlab_get_error
                    ),
                    atom.PunctualExceptionInformationForward(
                        exception_class=github.GithubException,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_github_error
                    )
                ]
            ),
            atom.AtomPairWithException(
                forward=self.run_analysis,
                forward_captured_exceptions=[]
            ),
        ]

    def handle_parse_exception(self, exception):
        pretty_result = "Parse Error: Unable to parse input."
        error_details = str(exception)
        return result.Result(
            integer_result=result.INTEGER_RESULT_INPUT_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": error_details}
        )

    def handle_gitlab_auth_error(self, exception):
        pretty_result = "GitLab Authentication Error: Unable to authenticate with GitLab."
        error_details = str(exception)
        return result.Result(
            integer_result=result.INTEGER_RESULT_TARGET_CONNECTION_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": error_details}
        )

    def handle_gitlab_get_error(self, exception):
        pretty_result = "GitLab Get Error: Unable to retrieve data from GitLab."
        error_details = str(exception)
        return result.Result(
            integer_result=result.INTEGER_RESULT_TARGET_CONNECTION_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": error_details}
        )

    def handle_github_error(self, exception):
        pretty_result = "GitHub Error: Unable to process GitHub request."
        error_details = str(exception)
        return result.Result(
            integer_result=result.INTEGER_RESULT_TARGET_CONNECTION_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": error_details}
        )

    def handle_artifact_exception(self, exception):
        pretty_result = f"Artifact Error: {str(exception)}"
        return result.Result(
            integer_result=result.INTEGER_RESULT_INPUT_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": "Artifact not found"}
        )

    def handle_label_column_exception(self, e):
        pretty_result = f"Configuration Error: {str(e)}"
        return result.Result(
            integer_result=result.INTEGER_RESULT_INPUT_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": "no label column found"}
        )

if __name__ == '__main__':
    entrypoint.start_execution(MostImportantVarProbe)
