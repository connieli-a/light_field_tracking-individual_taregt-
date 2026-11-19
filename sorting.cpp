#include "light_field_tracking.h"
#include <algorithm> 


using namespace std;
void extract_rows(const std::vector<CircleInf>& circleList, const float y_tolerance) {
    vector<CircleInf> sortedList;
    //sort by the y-axis
    sortedList = circleList;
    sort(sortedList.begin(), sortedList.end(),
        [](const CircleInf& a, const CircleInf& b) { return a.y < b.y; });
    rows.clear();
    for(auto&circle : sortedList){
		bool foundRow = false;

        for (auto& row : rows) {
			float meanY = 0.0f;

            for(const auto& c : row) {
                meanY += c.y;

			}
			meanY /= static_cast<float>(row.size());
            if (abs(meanY - circle.y) < y_tolerance) {
				row.push_back(circle);
                foundRow = true;
				break;
            }
        }
        if (!foundRow) {
			rows.push_back({ circle });
        }
	}
    
    for(auto& row : rows) {
        sort(row.begin(), row.end(), [](const CircleInf& a, const CircleInf& b) {
            return a.x < b.x;
        });
	}
    
     n_cols = static_cast<int>(rows[0].size());
     n_rows = static_cast<int>(rows.size());
    cout << "rows size" << rows.size() << endl;
    cout << "cols size" << n_cols << endl;
    
   
    //int estimated_rows = std::max(1, static_cast<int>(sortedList.size() / 30));
   
    //rows.reserve(estimated_rows);
    //while (!sortedList.empty())
    //{
    //    /* code */
    //    float row_y = sortedList[0].y;
    //    for (const auto& c : sortedList)
    //    {
    //        /* code */
    //        if (c.y < row_y)
    //            row_y = c.y;
    //    }
    //    vector<CircleInf>current_row;
    //    // current_row.reserve(34);
    //    vector<CircleInf>remaining_row;
    //    remaining_row.reserve(sortedList.size());
    //    for (const auto& c : sortedList) {
    //        if (abs(row_y - c.y) < y_tolerance)
    //            current_row.push_back(c);
    //        else remaining_row.push_back(c);
    //    }
    //    //sort by x-axis
    //    sort(current_row.begin(), current_row.end(), [](const CircleInf& a, const CircleInf& b) {
    //        return a.x < b.x;
    //        });
    //    rows.push_back(current_row);
    //    // cout<< current_row.size()<<endl;
    //    sortedList = move(remaining_row);
 
  
}

void prepare_data() {
    rows_flat.clear();
    rows_offsets.clear();
    //flatten the uploaded data
    int offsets = 0;
    for (auto& row : rows) {
        rows_offsets.push_back(offsets);
        rows_flat.insert(rows_flat.end(), row.begin(), row.end());
        offsets += row.size();
    }
    total_circles = static_cast<int>(rows_flat.size());

}

bool LightFieldTracking::acquire()
{
    return camera->grab(frame);
}

void LightFieldTracking::process(const cv::Mat& input) {
    lightFieldProcess.process(input);
}
cv::Mat LightFieldTracking::getImage() const{
    return frame;
}