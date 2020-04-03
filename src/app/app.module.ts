import { D3PackedBubbleChartService } from './bubble-chart/d3-packed-bubble-chart.service';
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BookComponent } from './book/book.component';
import { FormsModule } from '@angular/forms';

import {AutocompleteLibModule} from 'angular-ng-autocomplete';
import { BubbleChartComponent } from './bubble-chart/bubble-chart.component';
@NgModule({
  declarations: [
    AppComponent,
    BookComponent,
    BubbleChartComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    AutocompleteLibModule
  ],
  providers: [D3PackedBubbleChartService],
  bootstrap: [AppComponent]
})
export class AppModule { }
