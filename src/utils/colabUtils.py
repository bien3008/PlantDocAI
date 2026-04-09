import sys

def isColabEnvironment() -> bool:
    """Check if the code is running inside Google Colab."""
    return "google.colab" in sys.modules

def mountGoogleDrive(targetPath: str = "/content/drive") -> str:
    """
    Mount Google Drive if running in Colab.
    Returns the root path of the mounted drive.
    """
    if isColabEnvironment():
        try:
            from google.colab import drive
            print("[INFO] Google Colab detected. Mounting Google Drive...")
            drive.mount(targetPath)
            print(f"[INFO] Google Drive mounted at {targetPath}")
            return targetPath
        except Exception as e:
            print(f"[ERROR] Failed to mount Google Drive: {e}")
            return ""
    else:
        print("[INFO] Not running in Google Colab. Skipping Drive mount.")
        return ""

def setupColabOutput(outputDirStr: str, useGdrive: bool) -> str:
    """
    If running in Colab and requested, mount Drive and return a redirected output path.
    Otherwise, returns the original output path.
    """
    import os
    if useGdrive and isColabEnvironment():
        driveRoot = mountGoogleDrive()
        if driveRoot:
            gdrivePrefix = os.path.join(driveRoot, "MyDrive", "PlantDocAI_Outputs")
            baseDirName = os.path.basename(outputDirStr.rstrip("/"))
            if not baseDirName:
                baseDirName = "runs"
            redirectedPath = os.path.join(gdrivePrefix, baseDirName)
            print(f"[INFO] useGdrive flag enabled. Redirecting output to: {redirectedPath}")
            return redirectedPath
    return outputDirStr
