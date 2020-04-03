import { Component } from '@angular/core';
import { D3PackedBubbleChartService } from './d3-packed-bubble-chart.service';


@Component({
  selector: 'bubble-chart',
  templateUrl: './bubble-chart.component.html',
  styleUrls: ['./bubble-chart.component.css']
})
export class BubbleChartComponent {
  name = 'Angular 6';

  constructor(
    private d3PackedBubbleChartService: D3PackedBubbleChartService,
    /* private d3TreemapService: D3TreemapService */
  ) { }

  ngOnInit() {
    this.d3PackedBubbleChartService.renderChart();
    // this.d3TreemapService.renderChart();
  }
}
