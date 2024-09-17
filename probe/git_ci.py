from dataclasses import dataclass
from io import TextIOWrapper
import gitlab
import github
import zipfile

import requests

class gitCI:
    @dataclass
    class CIType:
        GITLAB: int = 0,
        GITHUB: int = 1

    def __init__(self,ci_type,**kwargs) -> None:
        """
        Configuration init
        @param ci_type: one of the possible values of CIType dataclass
        @param kwargs:
            - gl_domain: Gitlab Domain
            - gl_token: Gitlab Token
            - gl_project : Gitlab Project
            - gh_domain: Github Domain
            - gh_token: Github Token
            - gh_repo: Github Repository
        """
        if ci_type==gitCI.CIType.GITLAB:
            self.ci=gitlabCI(kwargs=kwargs)
        elif ci_type==gitCI.CIType.GITHUB:
            self.ci=githubCI(kwargs=kwargs)
        
    def getArtifact(self,**kwargs):
        """
        Get specific artifact
        @param kwargs:
            - branch_name: Branch name (Gitlab only)
            - job_name: Job name (Gitlab only)
            - artifact_name: Artifact name (Github only)
            - artifact_path : Artifact path
        """
        return self.ci.getArtifact(kwargs=kwargs)

class gitlabCI:
    def __init__(self,kwargs) -> None:
        self.gl=gitlab.Gitlab(url=kwargs.pop("gl_domain"), private_token=kwargs.pop("gl_token"))
        self.gl.auth()
        self.gl_project = self.gl.projects.get(kwargs.pop("gl_project"))
    
    def getArtifact(self,kwargs) -> TextIOWrapper:
        branch_name=kwargs.pop("branch_name")
        job_name=kwargs.pop("job_name")
        artifact_path=kwargs.pop("artifact_path")
        with open("/tmp/artifacts.zip", "wb") as artifacts_file:
            artifacts_bytes=self.gl_project.artifacts.download(ref_name=branch_name,job=job_name)
            artifacts_file.write(artifacts_bytes)
        with zipfile.ZipFile("/tmp/artifacts.zip","r") as zip_ref:
            zip_ref.extractall("/tmp/artifacts")
        return open("/tmp/artifacts/"+artifact_path, "r")
    
class githubCI:
    def __init__(self,kwargs) -> None:
        self.gh_token=kwargs.pop("gh_token")
        gh_auth=github.Auth.Token(token=self.gh_token)
        gh_domain=kwargs.pop("gh_domain")
        if gh_domain=="https://github.com" or gh_domain=="https://github.com/":
            self.gh=github.Github(auth=gh_auth)
        else:
            self.gh=github.Github(base_url=kwargs.pop("gh_domain")+"/api/v3",auth=gh_auth)
        self.gh_repo=self.gh.get_repo(kwargs.pop("gh_repo"))
    
    def getArtifact(self,kwargs) -> TextIOWrapper:
        artifact_name=kwargs.pop("artifact_name")
        artifact_path=kwargs.pop("artifact_path")
        with open("/tmp/artifacts.zip", "wb") as artifacts_file:
            artifact_list=self.gh_repo.get_artifacts(name=artifact_name).get_page(0)
            assert len(artifact_list)!=0, "Artifact doesn't exists"
            artifact_bytes=requests.get(url=self.gh_repo.get_artifacts(name=artifact_name).get_page(0)[0].archive_download_url, headers={'Authorization': f'token {self.gh_token}'}).content
            artifacts_file.write(artifact_bytes)
        with zipfile.ZipFile("/tmp/artifacts.zip","r") as zip_ref:
            zip_ref.extractall("/tmp/artifacts")
        # Simple translation for gitlab interoperability
        artifact_path_tokens=artifact_path.split("/")
        artifact_path=artifact_path_tokens[len(artifact_path_tokens)-1]
        return open("/tmp/artifacts/"+artifact_path, "r")
        
        
        
        
        
        
        
