#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <algorithm>

using namespace std;

struct Point {
    float x, y;
    int cluster = -1;
};


// Save points to CSV
void savePointsToCSV(const vector<Point>& points, const string& filename="data.csv") {
    ofstream out(filename);
    out << "x,y,cluster\n"; // header
    for (auto& p : points) {
        out << p.x << "," << p.y << "," << p.cluster << "\n";
    }
    out.close();
    cout << "Points saved to " << filename << endl;
}


// Euclidean distance
float distance(const Point& a, const Point& b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

// Generate random points
vector<Point> generatePoints(int n, float min=-50, float max=50) {
    vector<Point> points;
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        float x = min + static_cast<float>(rand()) / RAND_MAX * (max - min);
        float y = min + static_cast<float>(rand()) / RAND_MAX * (max - min);
        points.push_back({x, y});
    }
    return points;
}

// Save points to a text file
void savePointsToFile(const vector<Point>& points, const string& filename="data.txt") {
    ofstream out(filename);
    for (auto& p : points) {
        out << p.x << " " << p.y << endl;
    }
    out.close();
    cout << "Points saved to " << filename << endl;
}

// Load points from a text file
vector<Point> loadPointsFromFile(const string& filename="data.txt") {
    vector<Point> points;
    ifstream in(filename);
    if (!in) {
        cout << "File not found: " << filename << endl;
        return points;
    }
    Point p;
    while (in >> p.x >> p.y) {
        points.push_back(p);
    }
    in.close();
    cout << "Points loaded from " << filename << endl;
    return points;
}

// K-Means clustering
vector<Point> kMeans(vector<Point>& points, int k, float threshold=0.1f) {
    vector<Point> centers(points.begin(), points.begin() + k);
    random_shuffle(points.begin(), points.end());

    float maxMove;
    do {
        // Assign clusters
        for (auto& p : points) {
            float minDist = distance(p, centers[0]);
            int cluster = 0;
            for (int i = 1; i < k; ++i) {
                float d = distance(p, centers[i]);
                if (d < minDist) {
                    minDist = d;
                    cluster = i;
                }
            }
            p.cluster = cluster;
        }

        maxMove = 0;
        // Update centers
        for (int i = 0; i < k; ++i) {
            float sumX = 0, sumY = 0;
            int count = 0;
            for (auto& p : points) {
                if (p.cluster == i) {
                    sumX += p.x;
                    sumY += p.y;
                    count++;
                }
            }
            if (count > 0) {
                float newX = sumX / count;
                float newY = sumY / count;
                maxMove = max(maxMove, distance({newX, newY}, centers[i]));
                centers[i].x = newX;
                centers[i].y = newY;
            }
        }
    } while (maxMove > threshold);

    return centers;
}

// PPM of raw data
void outputPPMDataOnly(const vector<Point>& points, int width=500, int height=500) {
    ofstream img("data_only.ppm");
    img << "P3\n" << width << " " << height << "\n255\n";

    float minX = points[0].x, maxX = points[0].x;
    float minY = points[0].y, maxY = points[0].y;
    for (auto& p : points) {
        minX = min(minX, p.x); maxX = max(maxX, p.x);
        minY = min(minY, p.y); maxY = max(maxY, p.y);
    }

    auto scaleX = [width, minX, maxX](float x) { return static_cast<int>((x - minX) / (maxX - minX) * (width - 1)); };
    auto scaleY = [height, minY, maxY](float y) { return static_cast<int>((y - minY) / (maxY - minY) * (height - 1)); };

    vector<vector<int>> imgR(height, vector<int>(width, 200));
    vector<vector<int>> imgG(height, vector<int>(width, 200));
    vector<vector<int>> imgB(height, vector<int>(width, 200));

    int pointSize = 3;

    for (auto& p : points) {
        int x = scaleX(p.x);
        int y = height - 1 - scaleY(p.y);
        for (int dx = -pointSize/2; dx <= pointSize/2; dx++) {
            for (int dy = -pointSize/2; dy <= pointSize/2; dy++) {
                int px = x + dx, py = y + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    imgR[py][px] = 128;
                    imgG[py][px] = 128;
                    imgB[py][px] = 128;
                }
            }
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img << imgR[y][x] << " " << imgG[y][x] << " " << imgB[y][x] << " ";
        }
        img << "\n";
    }
    img.close();
    cout << "PPM image saved as data_only.ppm" << endl;
}

// PPM of clustered data
void outputPPM(const vector<Point>& points, const vector<Point>& centers, int width=500, int height=500) {
    ofstream img("clusters.ppm");
    img << "P3\n" << width << " " << height << "\n255\n";

    float minX = points[0].x, maxX = points[0].x;
    float minY = points[0].y, maxY = points[0].y;
    for (auto& p : points) {
        minX = min(minX, p.x); maxX = max(maxX, p.x);
        minY = min(minY, p.y); maxY = max(maxY, p.y);
    }

    auto scaleX = [width, minX, maxX](float x) { return static_cast<int>((x - minX) / (maxX - minX) * (width - 1)); };
    auto scaleY = [height, minY, maxY](float y) { return static_cast<int>((y - minY) / (maxY - minY) * (height - 1)); };

    int colors[12][3] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
        {255, 0, 255}, {0, 255, 255}, {128, 0, 0}, {0, 128, 0},
        {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128}
    };

    vector<vector<int>> imgR(height, vector<int>(width, 200));
    vector<vector<int>> imgG(height, vector<int>(width, 200));
    vector<vector<int>> imgB(height, vector<int>(width, 200));

    int pointSize = 3;
    int centerSize = 5;

    for (auto& p : points) {
        int x = scaleX(p.x);
        int y = height - 1 - scaleY(p.y);
        for (int dx = -pointSize/2; dx <= pointSize/2; dx++) {
            for (int dy = -pointSize/2; dy <= pointSize/2; dy++) {
                int px = x + dx, py = y + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    imgR[py][px] = colors[p.cluster % 12][0];
                    imgG[py][px] = colors[p.cluster % 12][1];
                    imgB[py][px] = colors[p.cluster % 12][2];
                }
            }
        }
    }

    for (int i = 0; i < centers.size(); i++) {
        int x = scaleX(centers[i].x);
        int y = height - 1 - scaleY(centers[i].y);
        for (int dx = -centerSize/2; dx <= centerSize/2; dx++) {
            for (int dy = -centerSize/2; dy <= centerSize/2; dy++) {
                int px = x + dx, py = y + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    imgR[py][px] = colors[i % 12][0];
                    imgG[py][px] = colors[i % 12][1];
                    imgB[py][px] = colors[i % 12][2];
                }
            }
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img << imgR[y][x] << " " << imgG[y][x] << " " << imgB[y][x] << " ";
        }
        img << "\n";
    }
    img.close();
    cout << "PPM image saved as clusters.ppm" << endl;
}


int main() {
    int nPoints = 5000;
    int k = 12;

    vector<Point> points;

    // Attempt to load points from file
    points = loadPointsFromFile("data.txt");

    // If file not found or empty, generate new points
    if (points.empty()) {
        points = generatePoints(nPoints);
        savePointsToFile(points, "data.txt");
    }

    outputPPMDataOnly(points);
    vector<Point> centers = kMeans(points, k);
    outputPPM(points, centers);
    savePointsToCSV(points);
    return 0;
}
