# This script will build the Docker image, save it, and push it to Docker Hub
# Make sure Docker Desktop is running before executing this!

$DockerHubUser = "adithiyasb"
$ImageName = "imagenet-classification-web-app"
$Tag = "latest"
$FullImageName = "$DockerHubUser/$ImageName`:$Tag"

Write-Host "1. Building the Docker image..."
docker-compose build

Write-Host "2. Tagging the image for Docker Hub..."
docker tag "imagenet-classification-web-app-web:latest" $FullImageName

Write-Host "3. Exporting the Docker container to a .tar archive..."
docker save -o "$ImageName.tar" "imagenet-classification-web-app-web:latest"

Write-Host "4. Compressing to .zip (and renaming to .rar as requested)..."
Compress-Archive -Path "$ImageName.tar" -DestinationPath "$ImageName.zip" -Force
Rename-Item -Path "$ImageName.zip" -NewName "$ImageName.rar" -Force
Remove-Item -Path "$ImageName.tar"

Write-Host "5. Push to Docker Hub (Manual Step)"
Write-Host "Please ensure you have run 'docker login' first!"
Write-Host "To push the image, run:"
Write-Host "docker push $FullImageName"

Write-Host "Done! You can now run the app using: docker-compose up -d"
