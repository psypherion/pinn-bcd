package main

import (
	"bytes"
	// "context"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	// "sync"
	"time"
)

// runPythonServer launches the Python model server as a background process.
func runPythonServer() error {
	// Adjust the command if using a virtual environment.
	cmd := exec.Command("python", "server/server_model.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start Python server: %v", err)
	}
	log.Printf("Python server started with PID %d", cmd.Process.Pid)
	// Wait a few seconds for the server to be ready.
	time.Sleep(5 * time.Second)
	return nil
}

// uploadToPythonServer forwards the image file to the Python model server's /predict endpoint.
func uploadToPythonServer(imagePath string) (string, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return "", fmt.Errorf("error opening image %s: %v", imagePath, err)
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", filepath.Base(imagePath))
	if err != nil {
		return "", fmt.Errorf("error creating form file for %s: %v", imagePath, err)
	}

	if _, err = io.Copy(part, file); err != nil {
		return "", fmt.Errorf("error copying file %s: %v", imagePath, err)
	}
	writer.Close()

	resp, err := http.Post("http://localhost:8001/predict", writer.FormDataContentType(), body)
	if err != nil {
		return "", fmt.Errorf("error sending request for %s: %v", imagePath, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response for %s: %v", imagePath, err)
	}
	return string(respBody), nil
}

// handler for /upload endpoint. This receives the file from the browser, forwards it to the Python API, and returns the result.
func uploadHandler(w http.ResponseWriter, r *http.Request) {
	// Parse multipart form.
	if err := r.ParseMultipartForm(10 << 20); err != nil {
		http.Error(w, "Unable to parse form", http.StatusBadRequest)
		return
	}
	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "File not provided", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Save the file temporarily.
	tempFile, err := os.CreateTemp("", "upload-*.jpg")
	if err != nil {
		http.Error(w, "Unable to create temp file", http.StatusInternalServerError)
		return
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()

	if _, err := io.Copy(tempFile, file); err != nil {
		http.Error(w, "Error saving file", http.StatusInternalServerError)
		return
	}

	// Forward the file to the Python server.
	result, err := uploadToPythonServer(tempFile.Name())
	if err != nil {
		http.Error(w, fmt.Sprintf("Error from Python server: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(result))
}

// serveStaticFiles serves files from the given directory.
func serveStaticFiles(prefix, dir string) {
	fs := http.FileServer(http.Dir(dir))
	http.Handle(prefix, http.StripPrefix(prefix, fs))
}

func main() {
	// Run the Python server in background.
	if err := runPythonServer(); err != nil {
		log.Fatalf("Error starting Python server: %v", err)
	}

	// Serve HTML and static assets.
	serveStaticFiles("/static/", "src")
	serveStaticFiles("/templates/", "templates")
	// Serve the index.html at the root.
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "templates/index.html")
	})

	// Set up /upload endpoint.
	http.HandleFunc("/upload", uploadHandler)

	port := 8080
	log.Printf("Go server starting on port %d...", port)
	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(port), nil))
}
